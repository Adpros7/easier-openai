import { b } from "./models.js";

const model = b.getModel(process.argv[2]);
let dep = model.data.deprecated;
console.log(
  JSON.stringify({
    deprecated: model.data.deprecated ?? false,
    modaltites: model.currentSnapshot.data.modalities,
    endpoints: model.currentSnapshot.data.supported_endpoints,
  }),
);
