import { b } from "./models.js";

const model = b.getModel("babbage-002");
console.log(model.currentSnapshot.data.supported_endpoints);
console.log(model.data.deprecated);
