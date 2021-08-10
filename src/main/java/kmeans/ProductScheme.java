package kmeans;

import au.com.bytecode.opencsv.CSVReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform;
import org.deeplearning4j.clustering.cluster.Point;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class ProductScheme {

    List<String>p;
    List<String>s;
    String[] missing;

    public String[] getMissing() {
        return missing;
    }

    public void setMissing(String[] missing) {
        this.missing = missing;
    }

    public List<String> getP() {
        return p;
    }

    public void setP(List<String> p) throws IOException, InterruptedException {
        this.p = p;
    }

    public List<String> getS() {
        return s;
    }

    public void setS(List<String> s) {

        this.s = s;
    }

    public  void main() throws IOException, InterruptedException {
        //CSVReader csvReader =  new CSVReader(new FileReader("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\predictionsfile1.csv"), ',');
        CSVReader csvReader = new CSVReader(new FileReader("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\taskassignmentN1.csv"), ',');

        // RecordReaderDataSetIterator
        List<String[]> s;
        s = csvReader.readAll();
        Iterator<String[]> it = s.iterator();
        Iterator<String[]> iterator = s.iterator();

        List<Point> points = new ArrayList<>();

        List<INDArray> vectors = new ArrayList<>();
        String[] columns = it.next();

        //DataSet alldata  = it.next();

        List<HashMap<String, Integer>> map
                = new ArrayList<>();
        HashMap<String, Integer> prodMap = new HashMap<>();
        HashMap<String, Integer> schemeMap = new HashMap<>();
        HashMap<String, Integer> amtMap = new HashMap<>();
        HashMap<String, Integer> reqTenueMap = new HashMap<>();
        HashMap<String, Integer> tenureMap = new HashMap<>();
        HashMap<String, Integer> reqRateMap = new HashMap<>();
        HashMap<String, Integer> effRateMap = new HashMap<>();
        HashMap<String, Integer> emiValMap = new HashMap<>();
        while (it.hasNext()) {
            String[] row = it.next();
             /*double[] doubleValues = Arrays.stream(row)
                     .mapToDouble(Double::parseDouble)
                     .toArray();*/
            if (!row[0].isEmpty()) {
                if (prodMap.containsKey(row[0])) {
                    prodMap.put(row[0], prodMap.get(row[0]) + 1);
                } else {
                    prodMap.put(row[0], 1);
                }
            }
            if (!row[1].isEmpty()) {
                if (schemeMap.containsKey(row[1])) {
                    schemeMap.put(row[1], schemeMap.get(row[1]) + 1);
                } else {
                    schemeMap.put(row[1], 1);
                }
            }
            if (!row[2].isEmpty()) {
                if (amtMap.containsKey(row[2])) {
                    amtMap.put(row[2], amtMap.get(row[2]) + 1);
                } else {
                    amtMap.put(row[2], 1);
                }
            }
            if (!row[3].isEmpty()) {
                if (reqTenueMap.containsKey(row[3])) {
                    reqTenueMap.put(row[3], reqTenueMap.get(row[3]) + 1);
                } else {
                    reqTenueMap.put(row[3], 1);
                }
            }
            if (!row[4].isEmpty()) {
                if (reqRateMap.containsKey(row[4])) {
                    reqRateMap.put(row[4], reqRateMap.get(row[4]) + 1);
                } else {
                    reqRateMap.put(row[4], 1);
                }
            }
            if (!row[5].isEmpty()) {
                if (effRateMap.containsKey(row[5])) {
                    effRateMap.put(row[5], effRateMap.get(row[5]) + 1);
                } else {
                    effRateMap.put(row[5], 1);
                }
            }
            if (!row[6].isEmpty()) {
                if (emiValMap.containsKey(row[6])) {
                    emiValMap.put(row[6], emiValMap.get(row[6]) + 1);
                } else {
                    emiValMap.put(row[6], 1);
                }
            }
            if (!row[7].isEmpty()) {
                if (tenureMap.containsKey(row[7])) {
                    tenureMap.put(row[7], tenureMap.get(row[7]) + 1);
                } else {
                    tenureMap.put(row[7], 1);
                }
            }
        }
        //   System.out.println(prodMap);
        String[] mode = new String[8];
        mode[0]= prodMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        mode[1] = schemeMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        mode[2] = amtMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        mode[3] = reqTenueMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        mode[4] = reqRateMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        mode[5] = effRateMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        mode[6] = emiValMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        mode[7] = tenureMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey();

        setMissing(mode);

        Set<String> setPro = prodMap.keySet();
        Set<String> setScheme = schemeMap.keySet();
        List<String> uniquePro = new ArrayList<>(setPro);
        List<String> uniqueScheme = new ArrayList<>(setScheme);

        setP(uniquePro);
        setS(uniqueScheme);
       /* System.out.println(uniquePro);
        System.out.println(uniquePro.size());
        System.out.println(uniqueScheme);
        System.out.println(uniqueScheme.size());*/
    }
}