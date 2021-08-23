package kmeans;

import org.datavec.api.transform.TransformProcess;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.File;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class UserData {

    public static void main(String[] args) throws Exception {
        //String[] arr ={"5000013","5000015","500000","24","17","10.5","2000","18"};

        ProductScheme ps = new ProductScheme();
        ps.main();
        HashMap<String,String>hm = ps.getColumnMode();
        LinkedHashMap<String,String>userData = new LinkedHashMap<>();
        userData.put("PRODUCT","5000013");
        userData.put("SCHEME","5000015");
        userData.put("LOAN_AMOUNT_REQUESTED","500000");
        userData.put("REQUESTED_TENURE","");
        userData.put("REQUESTED_RATE","");
        userData.put("EFFECTIVE_INTEREST_RATE","10.4");
        userData.put("EMI_BASE_VALUE","20000");
        userData.put("TENURE","18");

        for (Map.Entry<String,String> entry : userData.entrySet()){
            if(entry.getValue().isEmpty() ){
                //if(hm.containsKey(entry.getKey())){
                    userData.replace(entry.getKey(), hm.get(entry.getKey()));

                //}
            }
        }

        List<String> values = new ArrayList<>(userData.values());
        String[] arr =  new String[values.size()];
        for(int i=0;i<values.size();i++)
            arr[i]= values.get(i);
       // File fileTp = new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\tp.txt");
        String d = new String(Files.readAllBytes(Paths.get(("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\tp.txt"))));
        TransformProcess rd=TransformProcess.fromJson(d);
        Object[]tpData=rd.execute(rd.transformRawStringsToInput(arr)).toArray();
        double[] convertedArray = Arrays.stream(tpData) // converts to a stream
                .mapToDouble(num -> Double.parseDouble(num.toString())) // change each value to Double
                .toArray();

        float[] floats =new float[convertedArray.length];
        for(int i=0;i<convertedArray.length;i++)
            floats[i]=(float) convertedArray[i];
       /* Double[] val = new Double[tpData.length];
        for(int i=0;i<convertedArray.length;i++){
            val[i] = convertedArray[i];
        }*/
       /* Double[] inverse = Arrays.stream(convertedArray)
                .map(db -> Arrays.stream(db).toArray(Double[]::new))
                .toArray(Double[]::new);*/
        INDArray r = Nd4j.createFromArray(convertedArray);
        INDArray rfloat = Nd4j.createFromArray(floats);

        System.out.println(r.length());
        File file = new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\normalizer.txt");
        NormalizerSerializer loader = NormalizerSerializer.getDefault();
        DataNormalization restoredNormalizer = loader.restore(file);
        restoredNormalizer.transform(rfloat);


        String modelFile = "C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\centerPoints.txt";
        FileInputStream fileInput = new FileInputStream(new File(modelFile));
        ObjectInputStream objectInput = new ObjectInputStream(fileInput);
        List<Point>clt = (List<Point>) objectInput.readObject();
        objectInput.close();

        /*String normalizerFile = "C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\centerPoints.txt";
        FileInputStream fileInputNorm = new FileInputStream(new File(normalizerFile));
        ObjectInputStream objectInputNorm = new ObjectInputStream(fileInputNorm);
        DataNormalization normalization= (DataNormalization) objectInput.readObject();
        objectInputNorm.close();
        normalization.transform(rfloat);*/
       /* for(int i=0;i<model.size();i++){
            //System.out.println(model.get(i).length());
            double dis = Transforms.euclideanDistance(model.get(i),rfloat);
            System.out.println(dis);
        }*/
        //r.castTo(DataType.FLOAT);
        TreeMap<Double,String>map = new TreeMap<>();
        for(int i=0;i<clt.size();i++){

            Point p = clt.get(i);
            Double dis = Transforms.euclideanDistance(p.getArray(),rfloat);
            map.put(dis,p.getId());
            System.out.println(dis);

        }
        System.out.println(map.size());
        Map.Entry<Double,String> entry = map.entrySet().iterator().next();
        Double key = entry.getKey();
        String value = entry.getValue();
        System.out.println(key+" "+value);
        //System.out.println(model);
        //String csModel = new String(Files.readAllBytes(Paths.get(("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\csModel.txt"))));
        //System.out.println(csModel);

        //ClusterSet cs = (ClusterSet) csModel;
        //INDArray data = Nd4j.create(arr);
        //System.out.println(r);
        //restoredNormalizer.transform(r);

        //System.out.println(r);
    }



}
