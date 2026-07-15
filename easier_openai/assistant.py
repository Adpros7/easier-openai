from time import sleep
from openai.types.responses.response import Response
from httpx import URL
from openai.types.conversations.conversation import Conversation
from typing import Optional
from openai import OpenAI
from .models import Model


class Assistant:
    def __init__(
        self,
        instructions: str = "",
        api_key: Optional[str] = None,
        model: Model = "gpt-5.5",
        base_url: str = "https://api.openai.com/v1",
        conversation: bool = True,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.conversation: Conversation | None = (
            self.client.conversations.create() if conversation else None
        )
        self.model = model
        self.instructions: str = instructions

    def change_model(self, model: Model):
        self.model = model

    def new_conversation(self):
        self.conversation = self.client.conversations.create()

    def change_base_url(self, new_url: str):
        self.client._base_url = URL(new_url)

    def change_instructions(self, new_instructions: str):
        self.instructions: str = new_instructions

    class FailedError(RuntimeError):
        pass

    class Task:
        def __init__(self, client: OpenAI, id: str):
            self.openaiclient: OpenAI = client
            self.id = id

        def _get_resp(self) -> Response:
            return self.openaiclient.responses.retrieve(self.id)

        def check_progress(self):
            return self._get_resp().status

        def return_output_if_done(self, return_full_response: bool = False):
            if self._get_resp().status == "completed":
                return (
                    self._get_resp()
                    if return_full_response
                    else self._get_resp().output_text
                )

    def chat(
        self,
        input,
        long_running: bool = False,
        wait_for_finish: bool = True,
        return_full_response: bool = False,
        stream: bool = False,
    ):
        if not long_running:
            out: Response = self.client.responses.create(
                conversation=self.conversation.id if self.conversation else None,
                input=input,
                instructions=self.instructions,
                model=self.model,
            )

            return out if return_full_response else out.output_text

        else:
            if stream:
                raise NotImplementedError(
                    "Streaming support with long running taks (background mode) is not supported. Its not on the todo list either, but feel free to open a PR"
                )

            out: Response = self.client.responses.create(
                conversation=self.conversation.id if self.conversation else None,
                input=input,
                instructions=self.instructions,
                background=True,
                model=self.model,
            )

            if wait_for_finish:
                while out.status in {"queued", "in_progress"}:
                    sleep(2)
                    out: Response = self.client.responses.retrieve(out.id)

                if out.status in {"cancelled", "failed"}:
                    raise self.FailedError(f"Response Failed. {out.status}")

                if out.status == "incompleted":
                    raise self.FailedError(
                        f"Incomplete response. {out.incomplete_details}"
                    )

                return out if return_full_response else out.output_text

            else:
                return self.Task(self.client, out.id)


if __name__ == "__main__":
    bob = Assistant("you are a joke teller", model="gpt-5")
    stream = bob.chat("hi", long_running=True, wait_for_finish=False)
    for i in range(5):
        print(stream.check_progress())
        sleep(2)
        print(stream.return_output_if_done())
