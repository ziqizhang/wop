package uk.ac.shef.inf.wop.goldstandard;

import com.opencsv.*;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * TODO: data output needs to add product category label
 * <p>
 * this reads the annotation data file prepared by GS1AnnotationFileCreator, and can do several things
 * - take a sample of size X, equally distributed over all classes
 * - after the above file is annotated by a user, read it back in, and create another annotation file containing only
 * those annotated by the user, and output it
 * - take the original annotation file, and the sample file, create a disjoint set, and output them
 * in X splits equally distributed
 * - read back the user-annotated files, and join them together to create a new GS file
 * - read the original GS1 gold standard file, output it into another format containing only name, desc, and category
 */
public class GS1AnnotationUtil {

    public static void main(String[] args) throws IOException {
        // select a small sample for trial run annotation
        String annotationFile = "/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_for_annotation_WDC2018.csv";
        String outputFile = "/home/zz/Work/data/wop/goldstandard_eng_v1_utf8_for_annotation_WDC2018_sample.csv";
        int samplePerClass = 2;
        selectAnnotationSample(annotationFile,outputFile,samplePerClass);
        System.exit(0);

        // once the above sample is annotated, select the disjoint with the full annotation file
        // (i.e., find the parts that are not yet annotated)
        /*annotationFile = "/home/zz/Work/data/wop/swc/trial_annotation/goldstandard_eng_v1_utf8_for_annotation.csv";
        String annotatedSample = "/home/zz/Work/data/wop/swc/trial_annotation/" +
                "goldstandard_eng_v1_utf8_for_annotation_sample_ZZ.csv";
        String outputFolder = "/home/zz/Work/data/wop/swc/trial_annotation/splits";
        selectAnnotationDisjointSet(annotationFile, annotatedSample, outputFolder, 1000);*/


        String originalGSFile="/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv";
        String newGSFile="/home/zz/Work/data/wop/swc/trial_annotation/goldstandard_eng_v1_utf8.A.csv";
        convertOldGS2GS(originalGSFile, newGSFile);

        newGSFile="/home/zz/Work/data/wop/swc/trial_annotation/goldstandard_eng_v1_utf8.B.csv";
        String gs1TaxonomyFile = "/home/zz/Work/data/wdc/GS1_en_2019-06/GS1_en_2019-06.csv";
        annotationFile="/home/zz/Work/data/wop/swc/trial_annotation/goldstandard_eng_v1_utf8_for_annotation_sample_ZZ.csv";
        convertAnnotation2GS(annotationFile, originalGSFile,
                gs1TaxonomyFile, newGSFile);

    }

    public static CSVReader getCSVReader(String inFile, char separator) throws FileNotFoundException {
        FileInputStream fis = new FileInputStream(inFile);
        InputStreamReader isr = new InputStreamReader(fis,
                StandardCharsets.UTF_8);

        // create csvReader object passing
        // file reader as a parameter
        CSVParser parser = new CSVParserBuilder()
                .withSeparator(separator)
                .withIgnoreLeadingWhiteSpace(true)
                .withIgnoreQuotations(true)
                .build();

        CSVReader csvReader = new CSVReaderBuilder(isr)
                .withSkipLines(0)
                .withCSVParser(parser)
                .build();
        return csvReader;
    }

    public static CSVReader getCSVReader(String inFile) throws FileNotFoundException {
        return getCSVReader(inFile, ';');
    }

    public static CSVReader getCSVReaderC(String inFile, char separator) throws FileNotFoundException {
        return getCSVReader(inFile, separator);
    }

    public static CSVWriter getCSVWriter(String outFile) throws FileNotFoundException {
        FileOutputStream fos = new FileOutputStream(outFile);
        OutputStreamWriter osw = new OutputStreamWriter(fos,
                StandardCharsets.UTF_8);
        CSVWriter csvWriter = new CSVWriter(osw, ';', '"');
        return csvWriter;
    }

    /**
     * reads the annotation data file prepared by GS1AnnotationFileCreator,
     * take a sample of size X, equally distributed over all classes
     */
    public static void selectAnnotationSample(String annotationFile, String outputFile,
                                              int samplesPerClass) throws IOException {

        CSVReader csvReader = getCSVReader(annotationFile);
        CSVWriter csvWriter = getCSVWriter(outputFile);

        String[] nextRecord;

        int countLines = 0;
//        while ((nextRecord = csvReader.readNext()) != null) {
//            countLines++;
//        }
//        System.out.println(countLines);

        List<String[]> lines = new ArrayList<>();
        Map<String, List<Integer>> dataDistribution = new HashMap<>();
        Map<Integer, int[]> startEnd = new HashMap<>();

        String prod = null;
        int start = 0;
        while ((nextRecord = csvReader.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                csvWriter.writeNext(new String[]{"Original_GS_name", "url", "lvl3_label",
                        "name_similarity", "new_name", "label", "url", "desc", "id"});
                continue;
            }
            lines.add(nextRecord);

            String label = nextRecord[2];
            if (prod == null) {
                prod = nextRecord[0];
                start = countLines - 2;
            } else if (prod.equalsIgnoreCase(nextRecord[0]))
                continue;
            else {
                prod = nextRecord[0];
                startEnd.put(start, new int[]{start, countLines - 2});
                start = countLines - 2;
            }

            List<Integer> found = dataDistribution.get(label);
            if (found == null) {
                found = new ArrayList<>();
            }
            found.add(start);
            dataDistribution.put(label, found);

        }
        startEnd.put(start, new int[]{start, countLines});

        //now the sampling process and output
        Random rand = new Random();
        for (Map.Entry<String, List<Integer>> entry : dataDistribution.entrySet()) {
            String label = entry.getKey();
            List<Integer> found = entry.getValue();

            for (int i = 0; i < found.size() && i < samplesPerClass; i++) {
                int randomIndex = rand.nextInt(found.size());
                Integer randomElement = found.get(randomIndex);
                found.remove(randomIndex);

                //output to file
                int[] start_and_end = startEnd.get(randomElement);
                try {
                    for (int s = start_and_end[0]; s < start_and_end[1] && s<lines.size(); s++) {
                        String[] line = lines.get(s);
                        csvWriter.writeNext(line);
                    }
                }catch (IndexOutOfBoundsException e){
                    e.printStackTrace();
                }
            }
        }
        csvWriter.close();
    }

    /**
     * take the original annotation file, and the sample file, create a disjoint set, and output them
     * in X splits equally distributed
     */
    public static void selectAnnotationDisjointSet(String originalAnn, String annotatedFile,
                                                   String outFolder, int sizePerOutputFile) throws IOException {

        CSVReader csvReaderOriginal = getCSVReader(originalAnn);
        CSVReader csvReaderAnnotated = getCSVReader(annotatedFile);

        //read which products are annotated
        String[] nextRecord;

        int countLines = 0;
        Set<String> annotatedProducts = new HashSet<>();

        while ((nextRecord = csvReaderAnnotated.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                continue;
            }

            String name = nextRecord[0];
            annotatedProducts.add(name);
        }
        csvReaderOriginal.close();

        csvReaderOriginal = getCSVReader(originalAnn);
        int fileCounter = 1, prodCounter = 0, totalProds = 0;
        String prodName = null;
        originalAnn = new File(originalAnn).getName();
        originalAnn = originalAnn.substring(0, originalAnn.lastIndexOf("."));
        CSVWriter csvWriter = getCSVWriter(outFolder + "/" + originalAnn + ".part" + fileCounter + ".csv");
        countLines = 0;
        //read the original annotation prep file and output those that are not annotated
        while ((nextRecord = csvReaderOriginal.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                csvWriter.writeNext(new String[]{"Original_GS_name", "url", "lvl3_label",
                        "name_similarity", "new_name", "label", "url", "desc", "id"});
                continue;
            }

            String name = nextRecord[0];
            if (annotatedProducts.contains(name))
                continue;
            if (prodName == null) {
                prodCounter++;
                prodName = name;
            } else if (!prodName.equalsIgnoreCase(name)) {
                prodCounter++;
                prodName = name;
                if (prodCounter % sizePerOutputFile == 0) {
                    csvWriter.close();
                    fileCounter++;
                    totalProds += prodCounter;
                    prodCounter = 0;
                    csvWriter = getCSVWriter(outFolder + "/" + originalAnn + ".part" + fileCounter + ".csv");
                    csvWriter.writeNext(new String[]{"Original_GS_name", "url", "lvl3_label",
                            "name_similarity", "new_name", "label", "url", "desc", "id"});
                }
            }

            csvWriter.writeNext(nextRecord);

        }
        csvWriter.close();
        totalProds += prodCounter;
        System.out.println(String.format("Total annotated=%d, to annotate=%d", annotatedProducts.size(), totalProds
        ));
    }

    public static Map<String, String[]> readTaxonomyGS1(String gs1TaxonomyFile) throws IOException {
        CSVReader csvReaderAnnotated = getCSVReader(gs1TaxonomyFile,',');
        String[] nextRecord;

        int countLines = 0;
        Map<String, String[]> result = new HashMap<>();

        while ((nextRecord = csvReaderAnnotated.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                continue;
            }
            String lvl3 = nextRecord[2];
            String[] lvl2_1 = new String[2];
            lvl2_1[0] = nextRecord[1]; //lvl2
            lvl2_1[1] = nextRecord[0]; //lvl1

            result.put(lvl3, lvl2_1);
        }
        return result;
    }

    public static Map<String, String[]> readTaxonomyOriginalGS(String originalGSFile) throws IOException {
        CSVReader csvReaderAnnotated = getCSVReader(originalGSFile);
        String[] nextRecord;

        int countLines = 0;
        Map<String, String[]> result = new HashMap<>();

        while ((nextRecord = csvReaderAnnotated.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                continue;
            }
            String lvl3 = nextRecord[12];
            String[] lvl2_1 = new String[2];
            lvl2_1[0] = nextRecord[11]; //lvl2
            lvl2_1[1] = nextRecord[10];//lvl1

            result.put(lvl3, lvl2_1);
        }
        return result;
    }

    /**
     * read in the annotations by a user, and also the GS1 product classification taxonomy (from the original GS file
     * plus with modification by the GS1 taxonomy), output GS in the following format:
     * - url, name, category, desc, lvl1, lvl2, lvl3, id in the index, which index segment
     * read it back in, and create another annotation file containing only
     * those annotated by the user, and output it
     */
    public static void convertAnnotation2GS(String annotationFile, String originalGSFile,
                                            String gs1TaxonomyFile, String outputFile) throws IOException {
        Map<String, String[]> gs1Taxonomy = readTaxonomyGS1(gs1TaxonomyFile);
        Map<String, String[]> originalTaxonomy = readTaxonomyOriginalGS(originalGSFile);

        CSVReader csvReaderAnnotated = getCSVReader(annotationFile);
        String[] nextRecord;

        int countLines = 0;

        CSVWriter csvWriter = getCSVWriter(outputFile);
        csvWriter.writeNext(new String[]{"URL", "Name", "Site specific category", "Description",
                "Lvl1", "Lvl2", "Lvl3", "Solr index", "Record ID"});
        while ((nextRecord = csvReaderAnnotated.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                continue;
            }

            String lvl3 = nextRecord[4];
            if (lvl3.length() < 3)
                continue;
            String url = nextRecord[5];
            String cat = ""; //todo
            String desc = nextRecord[6];
            String name = nextRecord[3];
            String indexID = ""; //todo
            String recordID = nextRecord[7]; //
            String[] lvl2_1 = originalTaxonomy.get(lvl3);
            if (lvl2_1 == null)
                lvl2_1 = gs1Taxonomy.get(lvl3);
            if (lvl2_1==null) {
                System.out.println("skipped line "+countLines+" with lvl3="+lvl3);
                continue;
            }
            csvWriter.writeNext(new String[]{url, name, cat, desc, lvl2_1[1],
                    lvl2_1[0], lvl3, indexID, recordID});
        }
        csvReaderAnnotated.close();
        csvWriter.close();
    }

    /**
     * read in original GS by Muesel, output GS in the following format:
     * - url, name, category, desc, lvl1, lvl2, lvl3, id in the index, which index segment
     * read it back in, and create another annotation file containing only
     * those annotated by the user, and output it
     */
    public static void convertOldGS2GS(String originalGSFile,String outputFile) throws IOException {

        CSVReader csvReader = getCSVReader(originalGSFile);
        String[] nextRecord;

        int countLines = 0;

        CSVWriter csvWriter = getCSVWriter(outputFile);
        csvWriter.writeNext(new String[]{"URL", "Name", "Site specific category", "Description",
                "Lvl1", "Lvl2", "Lvl3", "Solr index", "Record ID"});
        while ((nextRecord = csvReader.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                continue;
            }

            String lvl3 = nextRecord[12];
            String lvl2=nextRecord[11];
            String lvl1=nextRecord[10];
            String url = nextRecord[2];
            String cat = nextRecord[8];
            String desc = nextRecord[5];
            String name = nextRecord[4];
            String indexID = "N/A";
            String recordID = "N/A";

            csvWriter.writeNext(new String[]{url, name, cat, desc, lvl1,
                    lvl2, lvl3, indexID, recordID});
        }
        csvReader.close();
        csvWriter.close();
    }
}
