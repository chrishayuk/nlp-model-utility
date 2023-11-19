import { NlpManager } from "node-nlp";
import fs from "fs";

interface TrainingDataItem {
    query: string;
    intent: string;
}

class NlpModelTrainer {
    // properties
    private modelFilePath: string;
    private trainingDataFile: string;

    // constructor
    constructor(modelFilePath: string, trainingDataFile: string) {
        // set the model path and training data file
        this.modelFilePath = modelFilePath;
        this.trainingDataFile = trainingDataFile;
    }

    // Asynchronous method to load training data
    private loadTrainingData(): TrainingDataItem[] {
        try {
            // read the file
            const rawData = fs.readFileSync(this.trainingDataFile, "utf8");

            // parse and return
            return JSON.parse(rawData) as TrainingDataItem[];
        } catch (error) {
            console.error("Error reading training data:", error);
            throw error;
        }
    }

    public async trainAndSaveModel(): Promise<NlpManager | null> {
        try {
            // load the training data
            const trainingData = this.loadTrainingData();

            // load the manager
            const manager = new NlpManager({ languages: ["en"], forceNER: true });

            // loop through the training data
            trainingData.forEach((item: { query: any; intent: any; }) => {
                // add each item as a document
                manager.addDocument('en', item.query, item.intent);
            });

            // train the model
            console.log('Training new model...');
            await manager.train();

            // write the model
            fs.writeFileSync(this.modelFilePath, manager.export(true));
            console.log('Model trained and saved successfully.');

            // return the manager
            return manager;
        } catch (error) {
            console.error("Error during model training:", error);
            throw error; // Rethrow to allow handling it in the calling function
        }
    }
}

export { NlpModelTrainer };
