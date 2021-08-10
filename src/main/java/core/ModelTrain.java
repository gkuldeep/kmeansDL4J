package core;

import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.KerasTokenizer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;

public class ModelTrain {

    private static Logger LOGGER = LoggerFactory.getLogger(ModelTrain.class);

    private final MultiLayerNetwork model;

    //word map
    private final KerasTokenizer tokenizer = new KerasTokenizer();

    //Train data
    private INDArray X;
    private INDArray Y;

    private int currentEpoch = 0;

    public ModelTrain(MultiLayerNetwork model, INDArray X, INDArray Y) {
        this.model = model;
        this.currentEpoch = model.getEpochCount();
        this.X = X;
        this.Y = Y;
    }

    public void trainModel(int toEpochCount, boolean persistModel) {
        if (this.X == null || this.Y == null) {
            //this.prepareTrainingData();
            LOGGER.warn("X or Y can't be null");
            return;
        }
        LOGGER.info("Starting model training");
        for (int epoch = this.currentEpoch + 1; epoch <= toEpochCount; epoch++) {
            //this.model.incrementEpochCount();
            this.currentEpoch = this.model.getEpochCount();
            this.model.fit(X, Y);
            this.model.computeGradientAndScore();
            LOGGER.info(String.format("Epoch : %s : Score : %f", epoch, this.model.score()));
            System.out.println(String.format("Epoch : %s : Score : %f", epoch, this.model.score()));
        }
        if (persistModel) {
            try {
                ModelSerializer.writeModel(this.model, "trainedModel_" + toEpochCount, true);
            } catch (IOException e) {
                LOGGER.info("Unable to save trained model");
            }
        }
        LOGGER.info("Model trained successfully");
    }

    public KerasTokenizer getTokenizer() {
        return this.tokenizer;
    }
}
