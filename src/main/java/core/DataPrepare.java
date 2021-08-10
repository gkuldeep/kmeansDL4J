package core;

import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.KerasTokenizer;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;

public class DataPrepare {
    private static Logger LOGGER = LoggerFactory.getLogger(DataPrepare.class);
    private final KerasTokenizer tokenizer = new KerasTokenizer();
    //Train data
    private INDArray X;
    private INDArray Y;
    //prepareTrainingData()
    public int prepareTrainingData() {

        //using class of nio file package
        Path filePath = Paths.get("src/main/resources/finndata.txt");
        //converting to UTF 8
        Charset charset = StandardCharsets.UTF_8;
        String texts = "";
        //try with resource
        try (BufferedReader bufferedReader = Files.newBufferedReader(filePath)) {
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                texts = texts.concat("\n" + line);
            }
        } catch (IOException ex) {
            LOGGER.warn("Text data load failed", ex.getMessage());
        }
        //cleaning text
        String textdata = texts.replaceAll("\n", " ");
        textdata = textdata.replaceAll("[^a-zA-Z ]", "").toLowerCase();
        String[] words = textdata.split(" ");

        //Tokenization
        this.tokenizer.fitOnTexts(words);
        saveTokenizertoJson(this.tokenizer);

        //WordIndexMap
        Map<String, Integer> wordIndexMap = this.tokenizer.getWordIndex();
        int vacab_size = wordIndexMap.size()+1;

        //sequence
        ArrayList<Integer> wordIndexSeq = new ArrayList<>();
        for (String word : words) {
            if (word != null && wordIndexMap.get(word) != null)
                wordIndexSeq.add(wordIndexMap.get(word));
        }
        //generating sequence list
        ArrayList<ArrayList<Integer>> sequences = new ArrayList<>();
        for (int i = 1; i < wordIndexSeq.size(); i++) {
            ArrayList<Integer> seq = new ArrayList<>();
            seq.add(wordIndexSeq.get(i - 1));
            seq.add(wordIndexSeq.get(i));
            if (!sequences.contains(seq))
                sequences.add(seq);
        }

        //Arrays of i/p and o/p data
        this.X = Nd4j.zeros(sequences.size(), 1);
        this.Y = Nd4j.zeros(sequences.size(), vacab_size);
        int index = 0;
        for (ArrayList<Integer> seq : sequences) {
            this.X.put(index, 0, seq.get(0));
            this.Y.put(index, seq.get(1), 1);
            index++;
        }

        return vacab_size;
    }

    private void saveTokenizertoJson(KerasTokenizer tokenizer) {
        BufferedWriter bufferedWriter = null;
        JSONObject jsonObject = new JSONObject();
        //config map for KerasTokenizer
        Map<String, Object> configMap = new HashMap<>();
        configMap.put("num_words", tokenizer.getNumWords());
        configMap.put("filters", tokenizer.getFilters());
        configMap.put("lower", true);
        configMap.put("split", " ");
        configMap.put("char_level", false);
        configMap.put("oov_token", tokenizer.getOutOfVocabularyToken());
        configMap.put("document_count", 1);
        configMap.put("word_counts", convertWithStream(tokenizer.getWordCounts()));
        configMap.put("word_docs", convertWithStream(tokenizer.getWordDocs()));
        configMap.put("index_docs", convertWithStream(tokenizer.getIndexDocs()));
        configMap.put("index_word", convertWithStream(tokenizer.getIndexWord()));
        configMap.put("word_index", convertWithStream(tokenizer.getWordIndex()));
        try {
            jsonObject.put("class_name", "Tokenizer");
            jsonObject.put("config", configMap);
        } catch (JSONException e) {
            LOGGER.warn("Json Conversion failed", e.getMessage());
        }
        ClassLoader classLoader = getClass().getClassLoader();
        File file = new File(classLoader.getResource(".").getFile() + "/word_dict.json");
        try {
            if (file.createNewFile()) {
                LOGGER.info("File is created!");
            } else {
                LOGGER.info("File already exists.");
            }
            bufferedWriter = new BufferedWriter(new FileWriter(file));
            bufferedWriter.write(String.valueOf(jsonObject));
            bufferedWriter.close();
            LOGGER.info("Json file saved");
        } catch (IOException e) {
            LOGGER.warn("Create json file failed", e.getMessage());
        }
    }

    public String convertWithStream(Map<?, ?> map) {
        String mapAsString = map.keySet().stream()
                .map(key ->  "\"" + key + "\"" + ": ".concat ((map.get(key) instanceof Integer) ? map.get(key)+"" : "\""+map.get(key)+"\"") )
                .collect(Collectors.joining(", ", "{", "}"));
        return mapAsString;
    }

    public INDArray getX() {
        return this.X;
    }

    public INDArray getY() {
        return this.Y;
    }
}
