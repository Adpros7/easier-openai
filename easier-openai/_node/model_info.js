import { b } from "./models.js";

const model = b.getModel(ProcessingInstruction.argv[2]);
console.log(JSON.stringify(model.data.deprecated));
