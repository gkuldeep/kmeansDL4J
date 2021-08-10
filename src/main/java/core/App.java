package core;

import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.KerasTokenizer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.Scanner;

public class App {
    public static void main(String[] args) {
        DataPrepare dataPrepare = new DataPrepare();
        int vocab_size = dataPrepare.prepareTrainingData();
        INDArray X = dataPrepare.getX();
        INDArray Y = dataPrepare.getY();
        ModelConfig modelConfig = new ModelConfig();
        MultiLayerNetwork model = modelConfig.getModel(vocab_size);

        ModelTrain modelTrain = new ModelTrain(model,X ,Y);
        modelTrain.trainModel(50,false);
        KerasTokenizer tokenizer = TokenizerFactory.loadKerasTokenizerFromPath("word_dict.json");
        Map<String, Integer> wordIndexMap = tokenizer.getWordIndex();
        Map<Integer, String> indexWordMap = tokenizer.getIndexWord();

        Inference inference = new Inference(model,wordIndexMap,indexWordMap);
        while (true){
            System.out.println("Enter Text: ");
            Scanner sc = new Scanner(System.in);
            String input= sc.nextLine();
            System.out.println(inference.getPrediction(input));
        }

    }
}
