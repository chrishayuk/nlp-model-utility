    import { NlpManager } from "node-nlp";
    import { NlpModelTrainer } from "./nlp-model-trainer";
    import fs from "fs";

    class NlpModelLoader {
        // properties
        private modelPath: string;

        // constructor
        constructor(path: string) {
            // set the path
            this.modelPath = path;
        }

        // checks if the model exists
        modelExists() : boolean {
            // returns if the model exists
            return fs.existsSync(this.modelPath);
        }

        // load or train the model
        async loadOrTrainModel(
            trainingDataPath: string,
            retrain: boolean
        ) : Promise<NlpManager | null> {
            // setup the model loader
            let manager;
        
            // check if we need to retrain the model, or the model doesn't exist
            if (retrain || !this.modelExists()) {
                // Load training data
                const trainer = new NlpModelTrainer(this.modelPath, trainingDataPath);
        
                // train and save the model
                manager = await trainer.trainAndSaveModel();
            } else {
                // load the model
                manager = await this.loadModel();
            }
        
            // ensure we have a model
            if (!manager) {
                throw new Error("Failed to load or train the NLP model.");
            }
        
            // return the model
            return manager;
        }

        // load the model
        async loadModel(): Promise<NlpManager | null> {
            // check if the model exists
            if (this.modelExists()) {
                try {
                    // attempt to load the model
                    console.log("Loading existing model...");
                    const data = fs.readFileSync(this.modelPath, "utf8");

                    // attempt to import
                    const manager = new NlpManager({ languages: ["en"], forceNER: true });
                    await manager.import(data);

                    // model loaded
                    console.log("Model loaded successfully.");

                    // return the model
                    return manager;
                } catch (error) {
                    // error loading the model
                    console.error("Error loading the model:", error);

                    // no model
                    return null;
                }
            } else {
                // log the issue
                console.log("No model found at the specified path.");

                // no model
                return null;
            }
        }
    }

    export {NlpModelLoader};
