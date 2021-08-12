package kmeans;

import au.com.bytecode.opencsv.CSVReader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.datavec.api.writable.NullWritable;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class DlfjKmeans  {



    public static void main(String[] args) throws IOException, InterruptedException {
        //CSVReader csvReader =  new CSVReader(new FileReader("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\predictionsfile1.csv"), ',');
        CSVReader csvReader =  new CSVReader(new FileReader("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\taskassignmentN1.csv"), ',');

        // RecordReaderDataSetIterator
        List<String[]> s;
         s=csvReader.readAll();
         Iterator<String[]> it=s.iterator();
        Iterator<String[]> iterator=s.iterator();
         //csvReader
         //it.next();

        //recordReaderTrain.asdasd
        List<Point>points=new ArrayList<>();

        Schema schema = new Schema.Builder()
                .addColumnCategorical("PRODUCT")
                .addColumnCategorical("SCHEME")
                .addColumnDouble("LOAN_AMOUNT_REQUESTED")
                .addColumnInteger("REQUESTED_TENURE")
                .addColumnDouble("REQUESTED_RATE")
                .addColumnDouble("EFFECTIVE_INTEREST_RATE")
                .addColumnDouble("EMI_BASE_VALUE")
                .addColumnInteger("TENURE")
                .build();



        TransformProcess tp = new TransformProcess.Builder(schema)
                .transform(new CategoricalToOneHotTransform("PRODUCT"))
                .transform(new CategoricalToOneHotTransform("SCHEME"))
                .build();
        Schema outputSchema = tp.getFinalSchema();
        List<INDArray>vectors = new ArrayList<>();
        String [] columns=it.next();

        //DataSet alldata  = it.next();

        List<HashMap<String, Integer> >map
                = new ArrayList<>();
        HashMap<String,Integer>prodMap=new HashMap<>();
        HashMap<String,Integer>schemeMap=new HashMap<>();
        HashMap<String,Integer>amtMap=new HashMap<>();
        HashMap<String,Integer>reqTenueMap=new HashMap<>();
        HashMap<String,Integer>tenureMap=new HashMap<>();
        HashMap<String,Integer>reqRateMap=new HashMap<>();
        HashMap<String,Integer>effRateMap=new HashMap<>();
        HashMap<String,Integer>emiValMap=new HashMap<>();
        while (it.hasNext()){
              String[] row =  it.next();
             /*double[] doubleValues = Arrays.stream(row)
                     .mapToDouble(Double::parseDouble)
                     .toArray();*/
            if(!row[0].isEmpty()) {
                if (prodMap.containsKey(row[0])) {
                    prodMap.put(row[0], prodMap.get(row[0]) + 1);
                } else {
                    prodMap.put(row[0], 1);
                }
            }
            if(!row[1].isEmpty()) {
                if (schemeMap.containsKey(row[1])) {
                    schemeMap.put(row[1], schemeMap.get(row[1]) + 1);
                } else {
                    schemeMap.put(row[1], 1);
                }
            }
            if(!row[2].isEmpty()) {
                if (amtMap.containsKey(row[2])) {
                    amtMap.put(row[2], amtMap.get(row[2]) + 1);
                } else {
                    amtMap.put(row[2], 1);
                }
            }
            if(!row[3].isEmpty()) {
                if (reqTenueMap.containsKey(row[3])) {
                    reqTenueMap.put(row[3], reqTenueMap.get(row[3]) + 1);
                } else {
                    reqTenueMap.put(row[3], 1);
                }
            }
            if(!row[4].isEmpty()) {
                if (reqRateMap.containsKey(row[4])) {
                    reqRateMap.put(row[4], reqRateMap.get(row[4]) + 1);
                } else {
                    reqRateMap.put(row[4], 1);
                }
            }
            if(!row[5].isEmpty()) {
                if (effRateMap.containsKey(row[5])) {
                    effRateMap.put(row[5], effRateMap.get(row[5]) + 1);
                } else {
                    effRateMap.put(row[5], 1);
                }
            }
            if(!row[6].isEmpty()) {
                if (emiValMap.containsKey(row[6])) {
                    emiValMap.put(row[6], emiValMap.get(row[6]) + 1);
                } else {
                    emiValMap.put(row[6], 1);
                }
            }
            if(!row[7].isEmpty()) {
                if (tenureMap.containsKey(row[7])) {
                    tenureMap.put(row[7], tenureMap.get(row[7]) + 1);
                } else {
                    tenureMap.put(row[7], 1);
                }
            }
           // System.out.println(prodMap);
            /*for(String st:row){
                HashMap<String,Integer>m = new HashMap<>();
                if (m.containsKey(st) ){
                    m.put(st, m.get(st) + 1);

                }
                else {
                    m.put(st, 1);
                }


            }*/
           /* List<Double> salaries = Arrays.stream(row)
                    .map(Double::valueOf)
                    .map(k -> k == null ? 0.0 : k.doubleValue())
                    .collect(Collectors.toList());*/

             /*Double[] doubleValues = Arrays.stream(row).map(k->{if(null!=k)return k;else return null;})
                     .map(Double::valueOf)
                     .toArray(Double[]::new);*/
            /*Double[] doubleValues = Arrays.stream(row)
                    .map(Double::valueOf)
                    .toArray(Double[]::new);*/
             //INDArray r = Nd4j.createFromArray(doubleValues);
             //vectors.add(r);
             /*for(int i=0;i< row.length;i++)
                System.out.print(row[i]+" ");
             System.out.println("\n");*/
        }
     //   System.out.println(prodMap);
        String[] mode = new String[8];
        String maxProd=prodMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[0]=maxProd;
        String maxScm=schemeMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[1]=maxScm;
        String maxAmt=amtMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[2]=maxAmt;
        String maxReqTenure=reqTenueMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[3]=maxReqTenure;
        String max=reqRateMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[4]=max;
        String maxEffRate=effRateMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[5]=maxEffRate;
        String maxemi=emiValMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[6]=maxemi;
        String maxTenure=tenureMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();
        mode[7]=maxTenure;
        iterator.next();
        while (iterator.hasNext()) {
            String[] row = iterator.next();
            for(int i=0;i<row.length;i++){
                if(row[i].isEmpty())
                    row[i]=mode[i];
            }
            Double[] doubleValues = Arrays.stream(row)
                    .map(Double::valueOf)
                    .toArray(Double[]::new);
            INDArray r = Nd4j.createFromArray(doubleValues);
            vectors.add(r);
        }


      //  System.out.println(vectors);
        double[][] features = new double[vectors.size()][];
        for(int i=0;i<vectors.size();i++){
            INDArray r = vectors.get(i);
            features[i]=r.toDoubleVector();
        }
        //features = (double[][]) vectors.toArray();
        //features[row] = getArrayOfObject(taskAssignSkObj);
        INDArray xfeatures = Nd4j.create(features);
        System.out.println(xfeatures.getColumn(0));
        for(int i=2;i<8;i++) {
            normalizeINDArray(xfeatures.getColumn(i));
            //System.out.println("-------");
            //System.out.println(xfeatures.getColumn(i));
        }

        Set<String>setPro=prodMap.keySet();
        Set<String>setScheme=schemeMap.keySet();
        List<String>uniquePro=new ArrayList<>(setPro);
        List<String>uniqueScheme=new ArrayList<>(setScheme);

        System.out.println(uniquePro);
        System.out.println(uniquePro.size());
        System.out.println(uniqueScheme);
        System.out.println(uniqueScheme.size());
       for(int i=0;i<8;i++) {

            System.out.println(xfeatures.getColumn(i));
            System.out.println("-------");
        }

        //normalizeINDArray(xfeatures.getColumn(1));
        //NullWritable n = new NullWritable();

        TransformProcess.Builder b = new TransformProcess.Builder(schema);
        //b.categoricalToOneHot(xfeatures.getColumn(0));
        xfeatures.getColumn(0);

        /*OneHot oneHot = new OneHot(xfeatures.getColumn(0),xfeatures.getColumn(0),0);
        System.out.println(oneHot);*/
       // oneHot.
       // Word2Vec w = new Word2Vec();
        //w.
        //TransformProcess.Builder n = new TransformProcess.Builder().categoricalToOneHot();
        System.out.println("-------------");
        points = Point.toPoints(xfeatures);

        //Point.toPoints()
        //System.out.println(xfeatures.getColumn(2));
        System.out.println(points.size());

        int maxIterationCount = 5;
        int clusterCount = 3;
        boolean useKplusplus = true;
        KMeansClustering kmc = KMeansClustering.setup(clusterCount,maxIterationCount, Distance.EUCLIDEAN,useKplusplus);
        ClusterSet cs = kmc.applyTo(points);

       /* for(int i=0;i<points.size();i++){
            Pair<Cluster, Double> c = cs.nearestCluster(points.get(i));
            Cluster cluster=c.getFirst();
            cluster.getCenter();
            System.out.println(cluster.getCenter().getArray()+" ---- "+c.getSecond());
            //System.out.println(cs.nearestCluster(points.get(i)).toString());
        }*/


        List<Cluster> clsterLst = cs.getClusters();


        Map<String,String> m=cs.getPointDistribution();

        /*for (Map.Entry<String,String> entry : m.entrySet())
            System.out.println("Key = " + entry.getKey() +
                    ", Value = " + entry.getValue());*/



        /*DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(points);
        normalizer.transform((INDArray) vectors);*/

        //for(int i=0;i<vectors.size();i++){
            System.out.println(vectors.get(0));
        //}

        //uniquePro= (List<String>) prodMap.keySet();

        System.out.println("\nCluster Centers:");
        List<INDArray>cen = new ArrayList<>();
        for(Cluster c: clsterLst) {

            Point center = c.getCenter();
            //System.out.println(center);
            //System.out.println("----");
            INDArray a = center.getArray();
            cen.add(a);
            System.out.println(center.getArray());

        }

        File f = new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers");




       ModelSerializer.addObjectToFile(f,"center",cen);
        //ModelSerializer.writeModel((Model) cen,f,true);
    }


    private static void normalizeINDArray(INDArray toNormalize) {
        INDArray columnMeans = toNormalize.mean(0);
        //INDArray columnMeans1 = toNormalize.mean();
        toNormalize.subiRowVector(columnMeans);
        INDArray std = toNormalize.std(0);
        //INDArray std1 = toNormalize.std();
        std.addi(Nd4j.scalar(1e-12));
        toNormalize.diviRowVector(std);

    }

}
