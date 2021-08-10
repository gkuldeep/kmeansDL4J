package core;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class ModelConfig {
    public MultiLayerNetwork getModel(int vacab_size){
        int numInputs = vacab_size;
        int numOutputs = vacab_size;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .cacheMode(CacheMode.DEVICE)
                .list()
                // layers in the network, added sequentially
                // parameters set per-layer override the parameters above
                .layer(0, new EmbeddingLayer.Builder().nIn(numInputs).nOut(150).build())
                .layer(1, new Bidirectional(new LSTM.Builder().units(120).activation(Activation.TANH).build()))
                .layer(2, new DenseLayer.Builder().units(240).activation(Activation.TANH).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.KL_DIVERGENCE)//Kullback Leibler Divergence Loss
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(240).nOut(numOutputs).build()).build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.setEpochCount(0);
        return network;
    }
}
