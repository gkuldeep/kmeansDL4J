package core;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public class Inference {
    private static Logger LOGGER = LoggerFactory.getLogger(Inference.class);
    private MultiLayerNetwork model;
    private Map<String, Integer> wordIndexMap;
    private Map<Integer, String> indexWordMap;

    public Inference(MultiLayerNetwork model, Map<String, Integer> wordIndexMap, Map<Integer, String> indexWordMap) {
        this.model = model;
        this.wordIndexMap = wordIndexMap;
        this.indexWordMap = indexWordMap;
    }

    public String getPrediction(String text) {
        //TODO: logging
        String[] texts = text.split(" ");
        text = texts[texts.length - 1];
        String nextWord = text;
        nextWord = getNextPrediction(nextWord, this.model, this.wordIndexMap,this.indexWordMap);
        return nextWord;
    }

    private String getNextPrediction(String text, MultiLayerNetwork model, Map<String, Integer> wordIndexMap, Map<Integer, String> indexWordMap) {
        //TODO: logging
        INDArray exp = Nd4j.zeros(1, 1);
        int[] pred = new int[2];
        if (wordIndexMap.get(text) != null) {
            exp.put(0, 0, Integer.parseInt(String.valueOf(wordIndexMap.get(text))));
            pred = model.predict(exp);
            for(int i=0;i<pred.length;i++)
                System.out.println(pred[i]);
        }
        return indexWordMap.get(String.valueOf(pred[0]));
    }

}
