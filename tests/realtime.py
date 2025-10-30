from __future__ import annotations

import asyncio
import base64
import logging
import threading
import wave
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, Optional, Union

import audioop  # type: ignore[attr-defined]
from openai import AsyncOpenAI, OpenAI

try:  # pragma: no cover - optional dependency
    import pyaudio  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pyaudio = None


logger = logging.getLogger(__name__)

# The OpenAI Python SDK exposes realtime connection helpers and webhook verification utilities. 
# 


@dataclass(slots=True)
class AudioProfile:
    '''PCM audio profile used for both capture and playback.'''

    sample_rate: int = 16_000
    sample_width: int = 2
    channels: int = 1
    chunk_duration_seconds: float = 0.2

    @property
    def frames_per_chunk(self) -> int:
        return int(self.sample_rate * self.chunk_duration_seconds)

    @property
    def bytes_per_frame(self) -> int:
        return self.sample_width * self.channels

    @property
    def bytes_per_chunk(self) -> int:
        return self.frames_per_chunk * self.bytes_per_frame


AudioInput = Union[str, Path, Any]


class _WebhookHTTPServer(ThreadingHTTPServer):
    '''HTTP server that routes verified webhook events into an asyncio queue.'''

    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type[BaseHTTPRequestHandler],
        *,
        loop: asyncio.AbstractEventLoop,
        async_queue: 'asyncio.Queue[dict[str, Any] | BaseException]',
        sync_client: OpenAI,
        path: str,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self.loop = loop
        self.async_queue = async_queue
        self.sync_client = sync_client
        self.expected_path = path


def _make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server: _WebhookHTTPServer  # type: ignore[assignment]

        def log_message(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - silence
            logger.debug('Webhook server log: %s', args)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != self.server.expected_path:
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get('Content-Length', '0'))
            payload = self.rfile.read(length).decode('utf-8')
            try:
                event = self.server.sync_client.webhooks.unwrap(payload, dict(self.headers))
            except Exception as exc:  # pragma: no cover - verification failures
            
                logger.exception('Invalid webhook signature')
                self.send_response(400)
                self.end_headers()
                self.wfile.write(str(exc).encode('utf-8'))
                return
            self.send_response(204)
            self.end_headers()
            event_dict = event.model_dump(mode='json')
            self.server.loop.call_soon_threadsafe(self.server.async_queue.put_nowait, event_dict)

    return Handler


class WebhookEventReceiver:
    '''Runs a lightweight HTTP server to collect webhook events asynchronously.'''

    _sentinel: dict[str, Any] = {'type': '__sentinel__'}

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        host: str,
        port: int,
        path: str,
        api_key: Optional[str],
        webhook_secret: str,
        url_override: Optional[str] = None,
    ) -> None:
        self.loop = loop
        self.host = host
        self.path = path if path.startswith('/') else f'/{path}'
        self._async_queue: 'asyncio.Queue[dict[str, Any] | BaseException]' = asyncio.Queue()
        self._server: Optional[_WebhookHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self.webhook_secret = webhook_secret
        self._url_override = url_override
        self._api_key = api_key
        self._port = port

    async def __aenter__(self) -> 'WebhookEventReceiver':
        self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._event_iterator()

    def start(self) -> None:
        if self._server is not None:
            
return
        sync_client = OpenAI(api_key=self._api_key, webhook_secret=self.webhook_secret)
        handler = _make_handler()
        self._server = _WebhookHTTPServer(
            (self.host, self._port),
            handler,
            loop=self.loop,
            async_queue=self._async_queue,
            sync_client=sync_client,
            path=self.path,
        )
        actual_host, actual_port = self._server.server_address
        self._resolved_url = (
            self._url_override
            if self._url_override
            else f'http://{actual_host if actual_host not in ('0.0.0.0', '::') else '127.0.0.1'}:{actual_port}{self.path}'
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.debug('Webhook server listening on %s', self._resolved_url)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.loop.call_soon_threadsafe(self._async_queue.put_nowait, self._sentinel)

    @property
    def url(self) -> str:
        if not hasattr(self, '_resolved_url'):
            raise RuntimeError('Webhook server has not been started')
        return self._resolved_url

    async def _event_iterator(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            item = await self._async_queue.get()
            if item is self._sentinel:
                break
            if isinstance(item, BaseException):
                raise item
            yield item


def _is_pyaudio_stream(source: Any) -> bool:
    return bool(pyaudio is not None and isinstance(source, pyaudio.Stream))


def _iter_audio_chunks(
    source: AudioInput,
    profile: AudioProfile,
    *,
    max_capture_seconds: Optional[float] = None,
) -> Iterable[bytes]:
    if _is_pyaudio_stream(source):
        frames_target = (
            int(max_capture_seconds * profile.sample_rate)
            
if max_capture_seconds is not None
            else None
        )
        frames_sent = 0
        while True:
            try:
                chunk = source.read(profile.frames_per_chunk, exception_on_overflow=False)
            except Exception:
                break
            if not chunk:
                break
            frames_sent += len(chunk) // profile.bytes_per_frame
            yield chunk
            if frames_target is not None and frames_sent >= frames_target:
                break
        return

    if hasattr(source, 'read') and callable(source.read):
        data = source.read()
    else:
        path = Path(str(source)).expanduser()
        with path.open('rb') as handle:
            data = handle.read()

    with wave.open(Path('input.wav'), 'rb') if False else wave.open(  # pragma: no cover - placeholder to satisfy type checker
        wave.BytesIO(data) if hasattr(wave, 'BytesIO') else wave.open  # type: ignore[misc]
    ):
        pass  # This block is never executed; retain placeholder to avoid static analysis issues.

    with wave.open(memoryview(data)) as wav_reader:
        frames = wav_reader.readframes(wav_reader.getnframes())
        sample_width = wav_reader.getsampwidth()
        channels = wav_reader.getnchannels()
        sample_rate = wav_reader.getframerate()

    mono = (
        audioop.tomono(frames, sample_width, 1, 1) if channels > 1 else frames
    )
    converted = (
        audioop.lin2lin(mono, sample_width, profile.sample_width)
        if sample_width != profile.sample_width
        else mono
    )
    resampled, _ = (
        audioop.ratecv(
            converted,
            profile.sample_width,
            1,
            sample_rate,
            profile.sample_rate,
            None,
        )
        if sample_rate != profile.sample_rate
        else (converted, None)
    )
    chunk_size = profile.bytes_per_chunk
    for offset in range(0, len(resampled), chunk_size):
        yield resampled[offset : offset + chunk_size]


def _encode_pcm_chunk(chunk: bytes) -> str:
    return base64.b64encode(chunk).decode('ascii')


def _extract_text(event: dict[str, Any]) -> list[str]:
    text_fragments: list[str] = []
    payload = event.get('data') or {}
    delta = payload.get('delta') or event.get('delta')

    def add_text(value: Any) -> None:
        if isinstance(value, str) and value:
            text_fragments.append(value)

    if isinstance(delta, dict):
        for key in ('text', 'output_text'):
            add_text(delta.get(key))
    elif isinstance(delta, str):
        add_text(delta)

    for key in ('text', 'output_text'):
        add_text(payload.get(key))

    segments = payload.get('segments')
    if isinstance(segments, list):
        for segment in segments:
            if isinstance(segment, dict):
                add_text(segment.get('text'))

    return text_fragments


def _extract_audio_bytes(event: dict[str, Any]) -> Optional[bytes]:
    payload = event.get('data') or {}
    delta = payload.get('delta') or event.get('delta')
    candidates = []
    if isinstance(delta, dict):
        candidates.extend(
            delta.get(key) for key in ('audio', 'chunk', 'data')
        )
    candidates.extend(payload.get(key) for key in ('audio', 'chunk'))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            try:
                return base64.b64decode(candidate)
            except Exception:
                continue
    return None


def _apply_session_overrides(base: dict[str, Any], overrides: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not overrides:
        return base
    merged = {**overrides}
    for key, value in base.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(value, dict) and isinstance(merged[key], dict):
            merged[key] = {**value, **merged[key]}
    return merged


def voice_to_text(
    source: AudioInput,
    *,
    model: str = 'gpt-4o-realtime-preview',
    api_key: Optional[str] = None,
    webhook_secret: str,
    webhook_host: str = '0.0.0.0',
    webhook_port: int = 0,
    webhook_path: str = '/openai-webhook',
    external_webhook_url: Optional[str] = None,
    audio_profile: AudioProfile = AudioProfile(),
    instructions: Optional[str] = None,
    session_overrides: Optional[dict[str, Any]] = None,
    max_capture_seconds: Optional[float] = None,
) -> Iterator[str]:
    '''Stream text deltas emitted via webhook while feeding audio into a realtime model.'''

    output_queue: 'queue.Queue[str | BaseException | object]' = queue.Queue()
    sentinel: object = object()

    async def runner() -> None:
        try:
            await _voice_to_text_async(
                source=source,
                model=model,
                api_key=api_key,
                webhook_secret=webhook_secret,
                webhook_host=webhook_host,
                webhook_port=webhook_port,
                
webhook_path=webhook_path,
                external_webhook_url=external_webhook_url,
                audio_profile=audio_profile,
                instructions=instructions,
              
  session_overrides=session_overrides,
                output_queue=output_queue,
            
    sentinel=sentinel,
                max_capture_seconds=max_capture_seconds,
            )
        except Exception as exc:
            logger.exception('voice_to_text encountered an error')
            output_queue.put(exc)
        finally:
            output_queue.put(sentinel)

    thread = threading.Thread(target=lambda: asyncio.run(runner()), daemon=True)
    thread.start()

    def iterator() -> Iterator[str]:
        try:
            while True:
                item = output_queue.get()
                if item is sentinel:
                  
  break
                if isinstance(item, BaseException):
                    raise item
                yield item  # type: ignore[misc]
        finally:
            thread.join(timeout=2)

    return iterator()


async def _voice_to_text_async(
    *,
    source: AudioInput,
    model: str,
    api_key: Optional[str],
    webhook_secret: str,
    webhook_host: str,
    webhook_port: int,
    webhook_path: str,
    external_webhook_url: Optional[str],
    audio_profile: AudioProfile,
    instructions: Optional[str],
    session_overrides: Optional[dict[str, Any]],
    output_queue: 'queue.Queue[str | BaseException | object]',
    sentinel: object,
    max_capture_seconds: Optional[float],
) -> None:
    loop = asyncio.get_running_loop()
    async with WebhookEventReceiver(
        loop=loop,
        host=webhook_host,
        port=webhook_port,
        path=webhook_path,
        api_key=api_key,
        webhook_secret=webhook_secret,
        url_override=external_webhook_url,
    ) as receiver:
        async_client = AsyncOpenAI(api_key=api_key)
        async with async_client.realtime.connect(model=model) as connection:
            session_payload: Dict[str, Any] = {
                
'modalities': ['audio', 'text'],
                'webhook': {
                    'url': receiver.url,
                    'secret': webhook_secret,
                },
            }
            if instructions:
                session_payload['instructions'] = instructions
            session_payload = _apply_session_overrides(session_payload, session_overrides)
            await connection.session.update(session=session_payload)

            async def consume_events() -> None:
                try:
                    async for event in receiver:
                        event_type = event.get('type', '')
                        if event_type.startswith('response'):
                            for fragment in _extract_text(event):
                                output_queue.put(fragment)
                           
 if event_type in {'response.completed', 'response.done'}:
                                
break
                        elif event_type == 'response.error':
                         
   message = event.get('data', {}).get('message', 'Unknown error')
                      
      raise RuntimeError(f'Realtime error: {message}')
                finally:
            
        output_queue.put(sentinel)

            consumer_task = asyncio.create_task(consume_events())

            for chunk in _iter_audio_chunks(source, audio_profile, max_capture_seconds=max_capture_seconds):
                await connection.input_audio_buffer.append(
          
          input_audio={'audio': _encode_pcm_chunk(chunk)}
                )
            await connection.input_audio_buffer.commit()
            await connection.response.create(response={'modalities': ['text']})
            await consumer_task


def voice_to_voice(
    source: AudioInput,
    *,
    model: str = 'gpt-4o-realtime-preview',
    api_key: Optional[str] = None,
    webhook_secret: str,
    webhook_host: str = '0.0.0.0',
    webhook_port: int = 0,
    webhook_path: str = '/openai-webhook',
    external_webhook_url: Optional[str] = None,
    audio_profile: AudioProfile = AudioProfile(),
    instructions: Optional[str] = None,
    session_overrides: Optional[dict[str, Any]] = None,
    voice: Optional[str] = None,
    max_capture_seconds: Optional[float] = None,
    playback_device_index: Optional[int] = None,
) -> None:
    '''Send audio from microphone or file to a realtime model and play the streamed response.'''

    asyncio.run(
        _voice_to_voice_async(
            source=source,
            model=model,
            api_key=api_key,
            webhook_secret=webhook_secret,
            webhook_host=webhook_host,
            webhook_port=webhook_port,
            webhook_path=webhook_path,
            external_webhook_url=external_webhook_url,
            audio_profile=audio_profile,
            instructions=instructions,
            session_overrides=session_overrides,
            voice=voice,
            max_capture_seconds=max_capture_seconds,
            playback_device_index=playback_device_index,
        )
    )


async def _voice_to_voice_async(
    *,
    source: AudioInput,
    model: str,
    api_key: Optional[str],
    webhook_secret: str,
    webhook_host: str,
    webhook_port: int,
    webhook_path: str,
    external_webhook_url: Optional[str],
    audio_profile: AudioProfile,
    instructions: Optional[str],
    session_overrides: Optional[dict[str, Any]],
    voice: Optional[str],
    max_capture_seconds: Optional[float],
    playback_device_index: Optional[int],
) -> None:
    if pyaudio is None:
        raise RuntimeError('PyAudio is required for voice playback; install pyaudio first.')

    pa_instance = pyaudio.PyAudio()
    output_stream = pa_instance.open(
        format=pa_instance.get_format_from_width(audio_profile.sample_width),
        channels=audio_profile.channels,
        rate=audio_profile.sample_rate,
        output=True,
        output_device_index=playback_device_index,
    )

    loop = asyncio.get_running_loop()
    async with WebhookEventReceiver(
        loop=loop,
        host=webhook_host,
        port=webhook_port,
        path=webhook_path,
        api_key=api_key,
        webhook_secret=webhook_secret,
        url_override=external_webhook_url,
    ) as receiver:
        async_client = AsyncOpenAI(api_key=api_key)
        async with async_client.realtime.connect(model=model) as connection:
            session_payload: Dict[str, Any] = {
          
      'modalities': ['audio', 'text'],
                'webhook': {
                  
  'url': receiver.url,
                    'secret': webhook_secret,
                },
            }
            if instructions:
                session_payload['instructions'] = instructions
            if voice:
                session_payload['voice'] = voice
            session_payload = _apply_session_overrides(session_payload, session_overrides)
           
 await connection.session.update(session=session_payload)

            async def consume_events() -> None:
                async for event in receiver:
                    event_type = event.get('type', '')
                    if event_type.startswith('response.audio.delta') or event_type.startswith('response.output_audio.delta'):
                        chunk = _extract_audio_bytes(event)
                        if chunk:
                            output_stream.write(chunk)
                    elif event_type.startswith('response'):
              
          if event_type == 'response.error':
                            message = event.get('data', {}).get('message', 'Unknown error')
                            raise RuntimeError(f'Realtime error: {message}')
                        if event_type in {'response.completed', 'response.done'}:
                            break

            consumer_task = asyncio.create_task(consume_events())

            for chunk in _iter_audio_chunks(source, audio_profile, max_capture_seconds=max_capture_seconds):
                await connection.input_audio_buffer.append(
                    input_audio={'audio': _encode_pcm_chunk(chunk)}
          
      )
            await connection.input_audio_buffer.commit()
            await connection.response.create(response={'modalities': ['audio', 'text']})

            try:
         
       await consumer_task
            finally:
                output_stream.stop_stream()
                output_stream.close()
                pa_instance.terminate()
"
