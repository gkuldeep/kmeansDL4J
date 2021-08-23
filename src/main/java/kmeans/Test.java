package kmeans;

import au.com.bytecode.opencsv.CSVReader;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.ClusterUtils;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Test {


    public static void main(String[] args) throws IOException {
        //CSVReader csvReader =  new CSVReader(new FileReader("C:\\Users\\posit\\Documents\\GitHub\\Kmeans\\predictionsfile1.csv"), ',');
        CSVReader csvReader =  new CSVReader(new FileReader("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\taskassignmentN1.csv"), ',');

        List<String[]> s;
        s=csvReader.readAll();
        Iterator<String[]> it=s.iterator();
        //it.next();

        //recordReaderTrain.
        List<Point>points=new ArrayList<>();

        List<INDArray>vectors = new ArrayList<>();
        it.next();
        List<Double[]> doubles=new ArrayList<>();
        int columns=0;
        while (it.hasNext()){
            String[] row =  it.next();
            // Double[] d = (Double[]) row;
             /*double[] doubleValues = Arrays.stream(row)
                     .mapToDouble(Double::parseDouble)
                     .toArray();*/
            Double[] doubleValues = Arrays.stream(row)/*.map(k-> {if(k.equals("")) {
                 k = null;
             }
                 return k;
             })*/
                    .map(y-> {
                        if(y.equals(""))
                        {
                            return null;
                        }
                        else {
                            return Double.valueOf(y);
                        }
                    })
                    .toArray(Double[]::new);
            doubles.add(doubleValues);
            if(columns==0)
            {
                columns=doubleValues.length;
            }
             INDArray r = Nd4j.createFromArray(doubleValues);
             vectors.add(r);
             /*for(int i=0;i< row.length;i++)
                System.out.print(row[i]+" ");
             System.out.println("\n");*/
        }
        Double avg[]=new Double[columns];
        for (int i=0;i<columns;i++)
        {
            avg[i]=0.0d;
        }

        for(Double[] d:doubles)
        {
            for (int i=0;i<columns;i++)
            {
                if(null!=d[i])
                {
                    avg[i]=avg[i]+d[i];
                }
            }
        }

        // C1 : LA -1 Cibil -1  Cluster ()
        // LA null Cibil null
        //Mode LA - use -- Cibil -- use Mode

        for (int i=0;i<columns;i++)
        {
            avg[i]=avg[i]/doubles.size();
        }
        System.out.println("-------------");
        points = Point.toPoints(vectors);

        System.out.println(points.size());

        int maxIterationCount = 5;
        int clusterCount = 3;
        boolean useKplusplus = true;
        //String distanceFunction = "cosinesimilarity";
        KMeansClustering kmc = KMeansClustering.setup(clusterCount,maxIterationCount, Distance.EUCLIDEAN,useKplusplus);
        //   Nd4j.getExecutioner().execAndReturn(ClusterUtils.createDistanceFunctionOp(Distance.EUCLIDEAN, m1.getArray(), m2.getArray())).getFinalResult().doubleValue();

        ClusterSet cs = kmc.applyTo(points);
        // cs.getCluster(123).addPoint();
        // ClusterUtils.refreshClustersCenters();
        Map<String,String> pointDistribution=cs.getPointDistribution();
       // List<KmeansDataHolder> dataHolders=new ArrayList<>();

       /* for(int i=0;i<points.size();i+=10){
            Pair<Cluster, Double> c = cs.nearestCluster(points.get(i));
            // c.getFirst()
            System.out.println(cs.nearestCluster(points.get(i)).toString());
        }*/


        List<Cluster> clsterLst = cs.getClusters();

        System.out.println("\nCluster Centers:");
        for(Cluster c: clsterLst) {
//

            Point center = c.getCenter();
            List<Point> pointsCluster=c.getPoints();

            Collections.sort(pointsCluster, new Comparator<Point>() {
                @Override
                public int compare(Point o1, Point o2) {
                    c.getCenter().getArray().castTo(DataType.FLOAT);
                    if(cs.getDistance(o1,o2)<c.getDistanceToCenter(o2))
                    {
                        return 1;
                    }
                    else if(c.getDistanceToCenter(o1)>c.getDistanceToCenter(o2))
                    {
                        return -1;
                    }
                    else
                    {
                        return 0;
                    }
                }
            });
            System.out.println(center);
            System.out.println("----");
            System.out.println(center.getArray());
           /* System.out.println("--label--");
            System.out.println(center.getLabel());
            System.out.println("---id-");
            System.out.println(center.getId());*/
        }


        //recordReaderTrain.initialize(new  FileSplit(File("src/main/resources/data/Data.csv")));
        // DataSetIterator dataIterTrain = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 3, 2);

    }




}