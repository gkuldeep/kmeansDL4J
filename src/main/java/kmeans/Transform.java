package kmeans;

import org.apache.commons.lang3.ObjectUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.ColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.NullWritable;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

public class Transform {

    public static void main(String[] args) throws IOException, InterruptedException {


       /* .addColumnDouble("LOAN_AMOUNT_REQUESTED",0.0,Double.MAX_VALUE,false,false)
                .addColumnInteger("REQUESTED_TENURE",0,Integer.MAX_VALUE)
                .addColumnDouble("REQUESTED_RATE",0.0,Double.MAX_VALUE,false,false)
                .addColumnDouble("EFFECTIVE_INTEREST_RATE",0.0,Double.MAX_VALUE,false,false)
                .addColumnDouble("EMI_BASE_VALUE",0.0,Double.MAX_VALUE,false,false)
                .addColumnInteger("TENURE",0,Integer.MAX_VALUE)*/
        ProductScheme ps = new ProductScheme();
        ps.main();
        List<String> prdt = ps.getP();
        List<String> scm = ps.getS();
        String[] mode = ps.getMissing();
        Schema schema = new Schema.Builder()
                .addColumnCategorical("PRODUCT",prdt)
                .addColumnCategorical("SCHEME",scm)
                //.addColumnString("",)
                /*.addColumnString("LOAN_AMOUNT_REQUESTED")
                .addColumnString("REQUESTED_TENURE")
                .addColumnString("REQUESTED_RATE")
                .addColumnString("EFFECTIVE_INTEREST_RATE")
                .addColumnString("EMI_BASE_VALUE")
                .addColumnString("TENURE")*/
                .addColumnDouble("LOAN_AMOUNT_REQUESTED",0.0,Double.MAX_VALUE,false,false)
                .addColumnInteger("REQUESTED_TENURE",0,Integer.MAX_VALUE)
                .addColumnDouble("REQUESTED_RATE",0.0,Double.MAX_VALUE,false,false)
                .addColumnDouble("EFFECTIVE_INTEREST_RATE",0.0,Double.MAX_VALUE,false,false)
                .addColumnDouble("EMI_BASE_VALUE",0.0,Double.MAX_VALUE,false,false)
                .addColumnInteger("TENURE",0,Integer.MAX_VALUE)
                .build();

        /*Writable w = new DoubleWritable(20);
        w.toDouble();*/
        //Condition

       /* .conditionalReplaceValueTransform("LOAN_AMOUNT_REQUESTED",new DoubleWritable(Double.parseDouble(mode[2])),new DoubleColumnCondition("LOAN_AMOUNT_REQUESTED", ConditionOp.Equal, null))
                .conditionalReplaceValueTransform("REQUESTED_TENURE",new IntWritable(Integer.parseInt(mode[3])),new IntegerColumnCondition("REQUESTED_TENURE", ConditionOp.Equal, null))
                .conditionalReplaceValueTransform("REQUESTED_RATE",new DoubleWritable(Double.parseDouble(mode[4])),new DoubleColumnCondition("REQUESTED_RATE", ConditionOp.Equal, null))
                .conditionalReplaceValueTransform("EFFECTIVE_INTEREST_RATE",new DoubleWritable(Double.parseDouble(mode[5])),new DoubleColumnCondition("EFFECTIVE_INTEREST_RATE", ConditionOp.Equal, null))
                .conditionalReplaceValueTransform("EMI_BASE_VALUE",new DoubleWritable(Double.parseDouble(mode[6])),new DoubleColumnCondition("EMI_BASE_VALUE", ConditionOp.Equal, null))



                .conditionalReplaceValueTransform("TENURE",new IntWritable(Integer.parseInt(mode[7])),new IntegerColumnCondition("TENURE", ConditionOp.Equal, null))
*/
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .categoricalToOneHot("PRODUCT")//Applying one-hot encoding
                .categoricalToOneHot("SCHEME")

                /*.conditionalReplaceValueTransform("LOAN_AMOUNT_REQUESTED",new DoubleWritable(Double.parseDouble(mode[2])),new DoubleColumnCondition("LOAN_AMOUNT_REQUESTED", ConditionOp.InSet, null))
                .conditionalReplaceValueTransform("REQUESTED_TENURE",new IntWritable(Integer.parseInt(mode[3])),new IntegerColumnCondition("REQUESTED_TENURE", ConditionOp.InSet, new HashSet<>(Arrays.asList())))
                .conditionalReplaceValueTransform("REQUESTED_RATE",new DoubleWritable(Double.parseDouble(mode[4])),new DoubleColumnCondition("REQUESTED_RATE", ConditionOp.InSet, new HashSet<>(Arrays.asList())))
                .conditionalReplaceValueTransform("EFFECTIVE_INTEREST_RATE",new DoubleWritable(Double.parseDouble(mode[5])),new DoubleColumnCondition("EFFECTIVE_INTEREST_RATE", ConditionOp.InSet, new HashSet<>(Arrays.asList())))
                .conditionalReplaceValueTransform("EMI_BASE_VALUE",new DoubleWritable(Double.parseDouble(mode[6])),new DoubleColumnCondition("EMI_BASE_VALUE", ConditionOp.InSet, new HashSet<>(Arrays.asList())))



                .conditionalReplaceValueTransform("TENURE",new IntWritable(Integer.parseInt(mode[7])),new IntegerColumnCondition("TENURE", ConditionOp.InSet, new HashSet<>(Arrays.asList())))

*/
                .build();

        String[] arr ={"5000013","5000015","500000","24","10","10.5","20000","15"};
        List<Writable> e=transformProcess.transformRawStringsToInput(arr);
        List<Writable> exe=transformProcess.execute(e);
        //INDArray p =exe.toArray();
         Object[] userData=  exe.toArray();
        double[] convertedArray = Arrays.stream(userData) // converts to a stream
                .mapToDouble(num -> Double.parseDouble(num.toString())) // change each value to Double
                .toArray();
        INDArray r = Nd4j.createFromArray(convertedArray);


        /*String p = "C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\tpJson.txt";
        StringBuilder outputToCSV = new StringBuilder();
        //outputToCSV.append("PRODUCT,SCHEME,LOAN_AMOUNT_REQUESTED,REQUESTED_TENURE,REQUESTED_RATE,EFFECTIVE_INTEREST_RATE,EMI_BASE_VALUE,TENURE\n");
        FileOutputStream writer = new FileOutputStream(new File(p));
        String tp=transformProcess.toJson();
        outputToCSV.append(tp);
        writer.write(outputToCSV.toString().getBytes("UTF-8"));*/
        String tp=transformProcess.toJson();
        FileWriter file1 = new FileWriter("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\tp.txt");
        file1.write(tp);
        file1.close();
        //INDArray xfeatures = Nd4j.create(userData);
        RecordReader reader = new CSVRecordReader(1,','); /* first line to skip and comma seperated */
        reader.initialize(new FileSplit(new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\file.csv")));
        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(reader,transformProcess);
        //Passing transformation process to convert the csv file


        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader, 7000, 7, 0);
        //iterator.next();


       // DataSet allData = iterator.next();
        //System.out.println(allData);


        DataNormalization dataNormalization = new NormalizerStandardize();
        dataNormalization.fit(iterator);
        iterator.setPreProcessor(dataNormalization);/*automatically perform transform on each iterator*/
        System.out.println(iterator);
        List<INDArray>vec = new ArrayList<>();
        DataSet ds=iterator.next();

        NormalizerSerializer saver = NormalizerSerializer.getDefault();
        File file = new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\normalizer.txt");
        saver.write(dataNormalization,file);
        INDArray a = ds.getFeatures();

       /* while (iterator.hasNext()){
            DataSet ds1=iterator.next();
            INDArray ia=ds1.getFeatures();
            vec.add(ia);
        }*/
       /* double[][] features = new double[vec.size()][];
        for(int i=0;i<vec.size();i++){
            INDArray r = vec.get(i);
            features[i]=r.toDoubleVector();
        }
        INDArray xfeatures = Nd4j.create(features);*/
       // INDArray xfeatures = Nd4j.create(vec);
        int maxIterationCount = 10;
        int clusterCount = 3;
        boolean useKplusplus = true;
        KMeansClustering kmc = KMeansClustering.setup(clusterCount,maxIterationCount, Distance.EUCLIDEAN,useKplusplus);
        List<Point>points = Point.toPoints(a);
        points.size();
        ClusterSet cs = kmc.applyTo(points);

       /* Point p= (Point) Point.toPoints(r);
        cs.nearestCluster(p);*/
        List<Cluster> clsterLst = cs.getClusters();

        for(Cluster c: clsterLst) {

            Point center = c.getCenter();
            //System.out.println(center);
            System.out.println("----");
            //INDArray a = center.getArray();
            System.out.println(center.getArray());

        }
       // dataNormalization.transform(r);
        System.out.println("end");
        //DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,10000,0.8);

        /*DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);
        System.out.println(allData);*/
    }

}

