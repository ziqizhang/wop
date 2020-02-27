package uk.ac.shef.inf.wop.goldstandard;

import com.google.gson.Gson;
import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * this class reads the annotated data files (by GS1AnnotationFileCreator) and convert them into the
 * json format for the swc2020 challenge
 *
 * Notice the following:
 * - it reads the original GS based on which the annotation files were extended, also a
 * 'correction' file to find if there are any corrected. If so, it corrects the original GS
 * and output json format
 * - it reads annotated files (format as GS1AnnotationFileCreator), output them
 * into json
 *
 * And it does not create train/val/test sets, this is done by Python (which can ensure
 * class distribution)
 */

public class SWC2020GSCreator {

    private Map<String,String> selectedProducts=new HashMap<>();
    private Map<String,String> selectedFrom=new HashMap<>();
    private Map<String, Integer> freq = new HashMap<>();
    private int id=0;

    public static void main(String[] args) throws IOException {
        SWC2020GSCreator sc = new SWC2020GSCreator();

        /*
        code for correcting and outputting original GS to json
         */
        String originalCSV="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv";
        String correctionCSV="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_correction.csv";
        String outFile="/home/zz/Work/data/wop/swc/swc_dataset/original.json";
        //sc.checkAndConvertOriginal(originalCSV, correctionCSV, outFile);

        /*
        code for converting annotations to json
         */
        String annotationFolder="/home/zz/Work/data/wop/swc/trial_annotation/Nov2017_omitNorms=false";
        outFile="/home/zz/Work/data/wop/swc/swc_dataset/extended.json";
        sc.convertAnnotations(annotationFolder, originalCSV, outFile);
    }

    /**
     * this deals task 1 above
     */
    public void checkAndConvertOriginal(String inputOriginalCSV,
                                               String inputCorrectionCSV,
                                               String outFile) throws IOException {
        int idCol=0, nameCol=4, descCol=5, urlCol=2, catCol1=8,catCol2=9, lvl1Col=10, lvl2Col=11, lvl3Col=12;

        Map<String, String[]> original = readToMap(inputOriginalCSV);
        Map<String, String[]> correction = readToMap(inputCorrectionCSV);
        List<Product> products = new ArrayList<>();

        for (Map.Entry<String, String[]> e : original.entrySet()){
            String id=e.getKey();
            String[] orig = e.getValue();
            String[] corr = correction.get(id);

            String lvl1=orig[lvl1Col], lvl2=orig[lvl2Col], lvl3=orig[lvl3Col];
            if (!orig[lvl3Col].equalsIgnoreCase(corr[lvl3Col])){
                System.out.println(String.format("Correction found for %s \t original=%s\t correction=%s",
                        orig[nameCol], orig[lvl3Col], corr[lvl3Col]));

                lvl1=corr[lvl1Col];
                lvl2=corr[lvl2Col];
                lvl3=corr[lvl3Col];
            }

            String cat=orig[catCol1];
            if (cat==null || cat.length()==0)
                cat=orig[catCol2];

            Product p = new Product(id, orig[nameCol].trim(), orig[descCol].trim(),
                    cat.trim(), orig[urlCol].trim(), lvl1,lvl2,lvl3);
            products.add(p);
        }

        Gson gson = new Gson();
        PrintWriter p = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8));
        for (Product pro: products){
            String js = gson.toJson(pro);
            p.println(js);
        }
        p.close();
    }

    private static Map<String, String[]> readToMap(String inputOriginalCSV) throws IOException {
        CSVReader csvReader = GS1AnnotationUtil.getCSVReader(inputOriginalCSV);
        String[] next;

        int countLines = 0;
//        while ((nextRecord = csvReader.readNext()) != null) {
//            countLines++;
//        }
//        System.out.println(countLines);

        Map<String, String[]> allRecords=new HashMap<>();
        while ((next = csvReader.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                continue;
            }
            String id = next[0];

            allRecords.put(id, next);
        }
        return allRecords;
    }

    /**
     * this deals task 2 above
     */
    public void convertAnnotations(String inputAnnotationFolder,
                                   String inputOriginalCSV,
                                        String outFile) throws IOException {
        Map<String, String[]> original = readToMap(inputOriginalCSV);
        Map<String, String> lvl3Tolvl2=new HashMap<>();
        Map<String, String> lvl3Tolvl1=new HashMap<>();
        int lvl1Col=10, lvl2Col=11, lvl3Col=12;
        for(String[] rec: original.values()){
            try {
                String lvl3 = rec[lvl3Col];
                String lvl2 = rec[lvl2Col];
                String lvl1 = rec[lvl1Col];
                lvl3Tolvl1.put(lvl3, lvl1);
                lvl3Tolvl2.put(lvl3, lvl2);
            }catch (Exception e){
                e.printStackTrace();
            }
        }

        PrintWriter p = new PrintWriter(new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8));

        File[] folders = new File(inputAnnotationFolder).listFiles();
        int totalFromFiles=0;
        for(File f: folders){
            if (!f.isDirectory())
                continue;

            char separator=';';
            if (f.getName().contains("comma"))
                separator=',';

            for (File af: f.listFiles()){
                if(!af.getName().endsWith("csv"))
                    continue;
                System.out.println(af);
                totalFromFiles+=process(af, p, separator, lvl3Tolvl1, lvl3Tolvl2);
            }
        }
        p.close();

        int sum=0;
        for (Map.Entry<String, Integer> e : freq.entrySet()){
            System.out.println(e.getKey()+"="+e.getValue());
            sum+=e.getValue();
        }
        System.out.println("\n"+sum+"/"+totalFromFiles);
        System.out.println("\tselected="+selectedProducts.size());
    }

    private int process(File f, PrintWriter p, char separator, Map<String, String> lvl3Tolvl1,
                         Map<String, String> lvl3Tolvl2) throws IOException {

        FileInputStream fis = new FileInputStream(f);
        InputStreamReader isr = new InputStreamReader(fis,
                StandardCharsets.UTF_8);

        // create csvReader object passing
        // file reader as a parameter
        CSVParser parser = new CSVParserBuilder()
                .withSeparator(separator)
                .withIgnoreLeadingWhiteSpace(true)
                .withQuoteChar('"')
                .build();

        CSVReader csvReader = new CSVReaderBuilder(isr)
                .withSkipLines(0)
                .withCSVParser(parser)
                .build();

        String[] nextRecord;
        Gson gson = new Gson();
        int nameCol=4, lvl3Col=5,catCol=6, urlCol=7, descCol=8;
        int count=0;

        if (f.toString().contains("batch4_semi/8.csv"))
            System.out.println();
        while ((nextRecord = csvReader.readNext()) != null) {
            String name=nextRecord[nameCol].trim();

            try {
                String cat = nextRecord[catCol].trim();
                String url = nextRecord[urlCol].trim();
                String desc = nextRecord[descCol].trim();
                String lvl3 = nextRecord[lvl3Col];
                if (lvl3 == null || lvl3.length() == 0)
                    continue;

                /*if (f.toString().contains("batch4_comma/36.csv") &&name.equalsIgnoreCase("Faux Fur Forrest Cushion"))
                    System.out.println();*/

                if (!lvl3.contains("_")) {
                    System.err.println("\t\terror:"+lvl3+"|\t\t"+name);
                    continue;
                }
                if (selectedProducts.containsKey(name)) {
                    String from=selectedFrom.get(name);
                    String annotation=selectedProducts.get(name);
                    if (annotation.equalsIgnoreCase(lvl3))
                        continue;
                    else{
                        System.err.println("\t\t\t inconsistent label for="+name+", A="+annotation+"\tB="+lvl3);
                    }
                }


                String lvl2 = lvl3Tolvl2.get(lvl3);
                String lvl1 = lvl3Tolvl1.get(lvl3);
                assert lvl2 != null && lvl1 != null;

                id++;
                Product pro = new Product(String.valueOf(id),
                        name, desc, cat, url, lvl1, lvl2, lvl3);
                String str = gson.toJson(pro);
                p.println(str);
                count++;
                selectedProducts.put(name,lvl3);
                selectedFrom.put(name, f.toString());

                Integer fr = freq.get(lvl3);
                if (fr == null) {
                    fr = 0;
                }
                fr++;
                freq.put(lvl3, fr);
            }catch (Exception e){
                e.printStackTrace();
            }

        }

        System.out.println("\ttotal="+count);;

        csvReader.close();

        return count;
    }

    private class Product {
        private String ID;
        private String Name;
        private String Description;
        private String CategoryText;
        private String URL;
        private String lvl1;
        private String lvl2;
        private String lvl3;

        public Product(String id, String name, String description, String categoryText,
                       String url, String lvl1, String lvl2, String lvl3) {
            this.ID = id;
            this.Name = name;
            this.Description = description;
            this.CategoryText=categoryText;
            this.URL=url;
            this.lvl1=lvl1;
            this.lvl2=lvl2;
            this.lvl3=lvl3;
        }

    }
}
