package core;

import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.preprocessing.text.KerasTokenizer;
import org.nd4j.common.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class TokenizerFactory {
    private static final Logger LOGGER = LoggerFactory.getLogger(TokenizerFactory.class);

    public static KerasTokenizer loadKerasTokenizerFromPath(String jsonPath){
        KerasTokenizer kerasTokenizer = null;
        try {
            kerasTokenizer = KerasTokenizer.fromJson(new ClassPathResource(jsonPath).getFile().getPath());
            LOGGER.info("Tokenizer loaded");
        } catch (IOException | InvalidKerasConfigurationException e) {
            LOGGER.warn("Failed to Load Tokenizer file",e.getMessage());
        }
        return kerasTokenizer;
    }
}
