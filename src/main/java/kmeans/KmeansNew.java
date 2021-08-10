package kmeans;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;

public class KmeansNew {
    public static void main(String[] args) throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(
                new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\taskassignmentN1.csv")));


        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, 150, 7, 2);
        //iterator.next();


        DataSet allData = iterator.next();
        //System.out.println(allData);


        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);
        System.out.println(allData);
    }
}
