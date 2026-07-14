import { b } from "./models.js";

const model = b.getModel(ProcessingInstruction.argv[2]);
console.log(model.data.deprecated);
