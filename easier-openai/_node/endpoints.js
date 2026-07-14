import { b } from "./models.js";

const model = b.getModel(process.argv[2]);
console.log(JSON.stringify(model.currentSnapshot.data.supported_endpoints));