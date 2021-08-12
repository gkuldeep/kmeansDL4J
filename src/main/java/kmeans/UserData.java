package kmeans;

import org.datavec.api.transform.TransformProcess;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class UserData {

    public static void main(String[] args) throws Exception {
        String[] arr ={"5000013","5000015","500000","24","10","10.5","20000","15"};


       // File fileTp = new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\tp.txt");
        String d = new String(Files.readAllBytes(Paths.get(("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\tp.txt"))));
        TransformProcess rd=TransformProcess.fromJson(d);
        Object[]tpData=rd.execute(rd.transformRawStringsToInput(arr)).toArray();
        double[] convertedArray = Arrays.stream(tpData) // converts to a stream
                .mapToDouble(num -> Double.parseDouble(num.toString())) // change each value to Double
                .toArray();
        INDArray r = Nd4j.createFromArray(convertedArray);

        System.out.println(tpData.length);
        File file = new File("C:\\Users\\gkuld\\OneDrive\\Desktop\\point\\Centers\\normalizer.txt");
        NormalizerSerializer loader = NormalizerSerializer.getDefault();
        DataNormalization restoredNormalizer = loader.restore(file);





        //INDArray data = Nd4j.create(arr);
        System.out.println(r);
        restoredNormalizer.transform(r);

        System.out.println(r);
    }



}
