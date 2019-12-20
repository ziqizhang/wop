package uk.ac.shef.inf.wop.goldstandard;

import com.opencsv.*;
import org.apache.commons.text.similarity.JaccardSimilarity;
import org.apache.commons.text.similarity.JaroWinklerSimilarity;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.FacetField;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.client.solrj.util.ClientUtils;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.wop.indexing.ProdCatDescIndexCreator;

import java.io.*;
import java.util.*;

/**
 * GS1 GS file prep:
 * - format:
 * lvl1, lvl2, lvl2
 * <p>
 * data file prep:
 * - format:
 * ref_prod name, ref_prod desc, ref_prod label; similarity, similar prod name, url, desc, label, id, which segment
 * ... (repeat 10)
 * - process:
 * for each prod in reference
 * search name in the 'desc' field of index
 * choose at most 50 different names
 * score similarity of names
 * rank by similarity asc
 * choose 10
 * <p>
 * Guidance:
 * - only annotate lvl3
 * - for each product, annotate at most 1
 * - always try to match the same label as the ref product, if
 * no match, annotate the product you are most confident with
 * <p>
 * TODO: summary of findings
 * <p>
 * The findings are reported on: index=2018; omitNorms=true
 * <p>
 * The following has been tried:
 * - query using name: works best
 * - query using desc: adds more noise. In fact too many words 'dilutes' focus of query
 * - query using name+cat (lvl3): same as above
 * - query using desc:name AND category:lvl3 cat : not particularly better than 1 or 3. perhaps same reason as 2
 * <p>
 * Each of the tried queries about also tested with:
 * - only filter results with a category (this brings worse results, but probably still we need this)1
 * - no restriction (this brings better results, but extremely sparse results have categories)
 * <p>
 * To try:
 * - all above with omitNorms=false
 * - all above on 2017 index
 */
public class GS1AnnotationFileCreator {


    /* PARAMS for creation from the original GS
        /home/zz/Work/data/wdc/WDCNov2017/prodcatdesc
prodcatdesc
/home/zz/Work/data/wop/goldstandard_eng_v1_utf8.csv
/home/zz/Work/data/wop/swc/trial_annotation/Nov2017_omitNorms=true

          PARAMS for analysing
    /home/zz/Work/data/wdc/WDCNov2018/prodcatdesc
    prodcatdesc
    /home/zz/Work/data/wop/swc/trial_annotation/goldstandard_eng_v1_utf8_for_annotation_sample.csv
    /home/zz/Work/data/wop/swc/trial_annotation/goldstandard_eng_v1_utf8_for_annotation_sample_2018cat.csv

         */
    public static void main(String[] args) throws IOException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient catdescIndex = new EmbeddedSolrServer(solrContainer.getCore(args[1]));
        GS1AnnotationFileCreator ic = new GS1AnnotationFileCreator(catdescIndex);
        //ic.process(args[2], args[3], 4, 2,12, 10);
        //ic.process(args[2], args[3], 0, -1, 1, 10);
        ic.process(args[2], args[3], 4, 2, 12, 10);
        catdescIndex.close();
        System.exit(0);
    }

    private static String FIELD_DESC = "desc";
    private static boolean USE_CATEGORY = true;
    private static int MAX_RESULTS_TO_CHECK = 100;
    private static int RECORDS_PER_BATCH = 100;
    private static final int MAX_DESC_WORDS = 500;

    private static double MIN_SIM_SCORE = 0.55;
    private static double MAX_SIM_SCORE = 0.95;
    private SolrClient prodCatDescIndex;
    //private JaroWinklerSimilarity stringSim = new JaroWinklerSimilarity();
    private JaccardSimilarity stringSim = new JaccardSimilarity();

    public GS1AnnotationFileCreator(SolrClient client) {
        this.prodCatDescIndex = client;
    }

    /**
     * @param originalGSFile
     * @param outputFolder
     * @param nameCol
     * @param lvl3Col
     * @throws IOException
     */
    private void process(String originalGSFile, String outputFolder,
                         int nameCol, int urlCol,
                         int lvl3Col, int topN) throws IOException {

        CSVReader csvReader = GS1AnnotationUtil.getCSVReader(originalGSFile);
        String[] next;

        int countLines = 0;
//        while ((nextRecord = csvReader.readNext()) != null) {
//            countLines++;
//        }
//        System.out.println(countLines);

        List<String[]> allRecords=new ArrayList<>();
        while ((next = csvReader.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                continue;
            }
            allRecords.add(next);
        }
        Collections.sort(allRecords, (t1, t2) -> {
            String l1 = t1[lvl3Col];
            String l2 = t2[lvl3Col];

            int c = l1.compareTo(l2);
            if (c==0){
                String n1=t1[nameCol];
                String n2=t2[nameCol];
                return n1.compareTo(n2);
            }
            return c;
        });

        CSVWriter csvWriter = GS1AnnotationUtil.getCSVWriter(outputFolder + "/1.csv");
        csvWriter.writeNext(new String[]{"Original_GS_name", //0
                "url",//1
                "lvl3_label",//2
                "name_similarity",//3
                "new_name",//4
                "label",//5
                "category",//6
                "url",//7
                "desc",
                "id"//8
        });

       countLines = 0;
//        while ((nextRecord = csvReader.readNext()) != null) {
//            countLines++;
//        }
//        System.out.println(countLines);

        String prevName = null;
        int maxDescWords = 0, sumDescWords = 0, fileCounter = 1, countprod=0;
        for (String[] nextRecord: allRecords) {
            countLines++;

            String name = nextRecord[nameCol];
            if (name.equalsIgnoreCase(prevName))
                continue;

            countprod++;
            String lvl3cat = nextRecord[lvl3Col];
            lvl3cat = lvl3cat.substring(lvl3cat.indexOf("_") + 1).replaceAll("[_/]", " ").trim();
            String desc = nextRecord[5];
            int len = desc.split("\\s+").length;
            sumDescWords += len;
            if (len > maxDescWords)
                maxDescWords = len;

            String value = name; // +" "+lvl3cat;
            if (value.length() == 0)
                value = name;

            String url = urlCol > -1 ? nextRecord[urlCol] : "none";
            System.out.println(String.format("currently line %d, name=%s", countLines, name));

            List<String[]> similar_withCat = null;
            if (USE_CATEGORY) {
                similar_withCat = getSimilarProducts(value, lvl3cat, topN, true);
            }
            List<String[]> similar_withOutCat = getSimilarProducts(value, lvl3cat, topN, false);


            /*
            - format:
 *      ref_prod name, ref_prod label; similarity, similar prod name, url, desc, label, id
 *                                                    ... (repeat 10)
             */

            if (similar_withCat == null && similar_withOutCat == null) {
                System.err.println(String.format("\t product %s has no matches:", name));
                continue;
            }

            if (similar_withCat != null) {
                for (String[] sim : similar_withCat) {
                    String[] line = new String[9];
                    line[0] = name;
                    line[1] = url;
                    line[2] = nextRecord[lvl3Col];
                    line[3] = "cat_"+sim[0];
                    line[4] = sim[1];
                    line[5] = "";
                    line[6] = sim[2];
                    line[7] = sim[3];
                    line[8] = sim[4];
                    csvWriter.writeNext(line);
                }
            }
            if (similar_withOutCat != null) {
                for (String[] sim : similar_withOutCat) {
                    String[] line = new String[9];
                    line[0] = name;
                    line[1] = url;
                    line[2] = nextRecord[lvl3Col];
                    line[3] = sim[0];
                    line[4] = sim[1];
                    line[5] = "";
                    line[6] = sim[2];
                    line[7] = sim[3];
                    line[8] = sim[4];
                    csvWriter.writeNext(line);
                }
            }

            prevName = name;

            if (countprod % RECORDS_PER_BATCH == 0) {
                csvWriter.close();
                fileCounter++;
                csvWriter = GS1AnnotationUtil.getCSVWriter(outputFolder + "/" + fileCounter + ".csv");
            }
        }
        csvReader.close();
        csvWriter.close();
        System.out.println("Total lines=" + countLines);
        System.out.println(maxDescWords);
        System.out.println((double) (sumDescWords / 8300));
    }

    /**
     * @param prodName
     * @return topN similar products, with the following values:
     * <p>
     * 0 - string similarity of names
     * 1 - name
     * 2 - source url
     * 3 - desc
     * 4 - id
     */
    private List<String[]> getSimilarProducts(String prodName, String prodCat, int topN,
                                              boolean useCatetogry) {
        //prodName = ProdCatDescIndexCreator.cleanName(prodName +" "+prodCat);
        prodName = ProdCatDescIndexCreator.cleanName(prodName);

        if (prodName == null)
            return null;

        SolrQuery query = new SolrQuery();

        if (!useCatetogry)
            query.setQuery(FIELD_DESC + ":(" + ClientUtils.escapeQueryChars(prodName) + ")");
        else {
            query.setQuery("category:* AND " + FIELD_DESC + ":("
                    + ClientUtils.escapeQueryChars(prodName) + ")");
        }

        query.setStart(0);
        query.setRows(5000);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        long count = 0;

        Set<String> uniqueNames = new HashSet<>();
        Map<SolrDocument, Double> similarities = new HashMap<>();

        try {
            res = prodCatDescIndex.query(query);
            if (res != null) {
                total = res.getResults().getNumFound();
                if (total > 0) {
                    for (SolrDocument doc : res.getResults()) {
                        String name = doc.getFieldValue("name").toString();
                        String desc = doc.getFieldValue(FIELD_DESC).toString();
                        if (desc.split("\\s+").length > MAX_DESC_WORDS)
                            continue;
                        name = ProdCatDescIndexCreator.cleanName(name);
                        if (name == null || uniqueNames.contains(name))
                            continue;

                        uniqueNames.add(name);
                        double score = stringSim.apply(prodName, name);
                        if (score < MIN_SIM_SCORE)
                            continue;
                        if (score > MAX_SIM_SCORE)
                            continue;
                        similarities.put(doc, score);
                        count++;

                        if (count > MAX_RESULTS_TO_CHECK)
                            break;
                    }
                }
            }

        } catch (Exception e) {
            e.printStackTrace();
        }


        List<SolrDocument> candidates = new ArrayList<>(similarities.keySet());

        candidates.sort((t1, t2) -> similarities.get(t2).compareTo(similarities.get(t1)));

        List<String[]> result = new ArrayList<>();
        for (int i = 0; i < candidates.size() && i < topN; i++) {
            SolrDocument doc = candidates.get(i);
            Double sim = similarities.get(doc);
            String[] values = new String[6];
            String id = doc.getFieldValue("id").toString();

            values[0] = String.valueOf(sim);
            values[1] = doc.getFieldValue("name").toString();
            //values[2] = doc.getFieldValue("url").toString();
            String url = id.split("\\|")[1].trim();
            values[3] = url;
            values[4] = doc.getFieldValue(FIELD_DESC).toString();
            values[2] = doc.getFieldValue("category").toString();
            values[5] = doc.getFieldValue("id").toString();
            result.add(values);
        }

        return result;
    }



}
