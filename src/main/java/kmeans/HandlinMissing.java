package kmeans;

import au.com.bytecode.opencsv.CSVReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.deeplearning4j.clustering.cluster.Point;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class HandlinMissing {

    public static void main(String[] args) throws IOException {
        CSVReader csvReader =  new CSVReader(new FileReader("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\taskassignmentN1.csv"), ',');

        // RecordReaderDataSetIterator
        List<String[]> s;
        s=csvReader.readAll();

        Map<String,ArrayList<?>>map = new HashMap<>();
        Iterator<String[]> it=s.iterator();

        List<Point>points=new ArrayList<>();

        List<INDArray>vectors = new ArrayList<>();
        String [] col=it.next();

        List<String[]>list=new ArrayList<>();
        List<String>prod = new ArrayList<>();
        //DataSet alldata  = it.next();

        while (it.hasNext()){
           /* for(int i=0;i<col.length;i++){
                String[] row =  it.next();
                ArrayList<?>l = new ArrayList<>();
                //l.add(row[i]);
                map.put(col[i],l);
            }*/
            String[] row =  it.next();
             /*double[] doubleValues = Arrays.stream(row)
                     .mapToDouble(Double::parseDouble)
                     .toArray();*/

            //computeMode(r);
            list.add(row);
             for(int i=0;i< row.length;i++)
                System.out.print(row[i]+" ");
             System.out.println("\n");
        }


       // list.get(0);
        double[][] features = new double[list.size()][];
        for(int i=0;i<list.size();i++){
            String[] str=list.get(i);
            for(int j=0;j< str.length;j++){
                if(str[j]!=null){
                    features[i][j]= Double.parseDouble(str[j]);
                }
                features[i][j]=-1;
                System.out.println(features[i][j]);
            }
            System.out.println("\n");

        }

        /*double[][] features = new double[vectors.size()][];
        for(int i=0;i<vectors.size();i++){
            INDArray r = vectors.get(i);
            features[i]=r.toDoubleVector();
        }
        //features = (double[][]) vectors.toArray();
        //features[row] = getArrayOfObject(taskAssignSkObj);
        INDArray xfeatures = Nd4j.create(features);
        System.out.println(xfeatures.getColumn(0));*/
    }
}
