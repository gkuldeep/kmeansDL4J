package kmeans;

import org.apache.commons.lang3.ObjectUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.ColumnCondition;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.condition.column.IntegerColumnCondition;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
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
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.*;

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
        //HashMap<String,String>hm = ps.getColumnMode();
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
                .removeColumns("PRODUCT[5000161]")
                .removeColumns("SCHEME[5000131]")

                .build();

        /*String[] arr ={"5000013","5000015","500000","24","10","10.5","20000","15"};
        List<Writable> e=transformProcess.transformRawStringsToInput(arr);
        List<Writable> exe=transformProcess.execute(e);
        Object[] userData=  exe.toArray();
        double[] convertedArray = Arrays.stream(userData) // converts to a stream
                .mapToDouble(num -> Double.parseDouble(num.toString())) // change each value to Double
                .toArray();
        INDArray r = Nd4j.createFromArray(convertedArray);*/


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


        DataSetIterator iterator = new RecordReaderDataSetIterator(transformProcessRecordReader, 7000);
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
        //dataNormalization.

        String path = "C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\NormalizerFile.txt";
        StringBuilder outputToCSV = new StringBuilder();
        //outputToCSV.append("PRODUCT,SCHEME,LOAN_AMOUNT_REQUESTED,REQUESTED_TENURE,REQUESTED_RATE,EFFECTIVE_INTEREST_RATE,EMI_BASE_VALUE,TENURE\n");
        outputToCSV.append(dataNormalization);
        FileOutputStream writer = new FileOutputStream(new File(path));
        writer.write(outputToCSV.toString().getBytes("UTF-8"));

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
        //points.ad
        ClusterSet cs = kmc.applyTo(points);



        //String csModel=cs.toString();
       /* Point p= (Point) Point.toPoints(r);
        cs.nearestCluster(p);*/
        List<Cluster> clsterLst = cs.getClusters();

       /* FileWriter csfile = new FileWriter("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\csModel.txt");
        csfile.write(csModel);
        csfile.close();*/

        List<INDArray>cen = new ArrayList<>();
        List<Point>centerPoints = new ArrayList<>();
        for(Cluster c: clsterLst) {

            Point center = c.getCenter();
            List<Point> allPoint=c.getPoints();
            for(Point p:allPoint){
                //p.getArray()
            }

           // centerPoints.add(center);
            System.out.println("----");
            //INDArray a = center.getArray();
            cen.add(center.getArray());
            System.out.println(center.getId());

        }
       /* String p = "C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\centroids.txt";
        FileOutputStream fileOutputStream = new FileOutputStream(p);
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(fileOutputStream);
        objectOutputStream.writeObject(cen);
        objectOutputStream.close();*/

        String p1 = "C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\centerPoints.txt";
        FileOutputStream fileOutputStream1 = new FileOutputStream(p1);
        ObjectOutputStream objectOutputStream1 = new ObjectOutputStream(fileOutputStream1);
        objectOutputStream1.writeObject(centerPoints);
        objectOutputStream1.close();

        /*StringBuilder outputToCSV = new StringBuilder();
        FileOutputStream writer = new FileOutputStream(new File(p));
        outputToCSV.append(cen);
        writer.write(outputToCSV.toString().getBytes("UTF-8"));*/
       // dataNormalization.transform(r);

        for(Cluster c: clsterLst) {
//

            //Point center = c.getCenter();
            List<Point> pointsCluster = c.getPoints();

            System.out.println(pointsCluster.size());

            Collections.sort(pointsCluster, (o1, o2) -> {
                c.getCenter().getArray().castTo(DataType.FLOAT);
                if (c.getDistanceToCenter(o1) < c.getDistanceToCenter(o2)) {
                    return 1;
                } else if (c.getDistanceToCenter(o1) > c.getDistanceToCenter(o2)) {
                    return -1;
                } else {
                    return 0;
                }
            });
            List<Point> pointsCluster1 = c.getPoints();
        }

        HashMap<String,List<Point>>hm = new HashMap<>();
        for(Cluster c:clsterLst){
            List<Point> pointsCluster1 = c.getPoints();
            hm.put(c.getId(),pointsCluster1);

        }


        Map<String,String> m=cs.getPointDistribution();

       /* for (Map.Entry<String,String> entry : m.entrySet())
            System.out.println("Key = " + entry.getKey() +
                    ", Value = " + entry.getValue());*/
        System.out.println("end");
        //DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,10000,0.8);

        /*DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(allData);
        normalizer.transform(allData);
        System.out.println(allData);*/
    }

}

