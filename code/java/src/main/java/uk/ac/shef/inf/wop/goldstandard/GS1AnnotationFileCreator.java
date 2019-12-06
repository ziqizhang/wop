package uk.ac.shef.inf.wop.goldstandard;

import com.opencsv.*;
import org.apache.commons.text.similarity.JaccardSimilarity;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.client.solrj.util.ClientUtils;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.wop.indexing.ProdCatDescIndexCreator;

import java.io.*;
import java.util.*;

/**
 * TODO: data output needs to add product category label, and which index segment
 *
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
 */
public class GS1AnnotationFileCreator {

    public static void main(String[] args) throws IOException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient catdescIndex = new EmbeddedSolrServer(solrContainer.getCore(args[1]));
        GS1AnnotationFileCreator ic = new GS1AnnotationFileCreator(catdescIndex);
        ic.process(args[2], args[3], 4, 2,12, 10);
        catdescIndex.close();
        System.exit(0);
    }

    private static int MAX_RESULTS_TO_CHECK = 100;
    private static double MIN_SIM_SCORE=0.5;
    private static double MAX_SIM_SCORE=0.9;
    private SolrClient prodCatDescIndex;
    private JaccardSimilarity stringSim = new JaccardSimilarity();

    public GS1AnnotationFileCreator(SolrClient client) {
        this.prodCatDescIndex = client;
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
    private List<String[]> getSimilarProducts(String prodName, int topN) {
        prodName = ProdCatDescIndexCreator.cleanName(prodName);
        if (prodName == null)
            return null;

        SolrQuery query = new SolrQuery();

        query.setQuery("text:" + ClientUtils.escapeQueryChars(prodName));
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
                        name = ProdCatDescIndexCreator.cleanName(name);
                        if (name == null || uniqueNames.contains(name))
                            continue;

                        uniqueNames.add(name);
                        double score = stringSim.apply(prodName, name);
                        if (score<MIN_SIM_SCORE)
                            continue;
                        if (score>MAX_SIM_SCORE)
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
            String[] values = new String[5];
            String id = doc.getFieldValue("id").toString();

            values[0] = String.valueOf(sim);
            values[1] = doc.getFieldValue("name").toString();
            //values[2] = doc.getFieldValue("url").toString();
            String url = id.split("\\|")[1].trim();
            values[2]=url;
            values[3] = doc.getFieldValue("text").toString();
            values[4] = doc.getFieldValue("id").toString();
            result.add(values);
        }

        return result;
    }

    /**
     * @param originalGSFile
     * @param outputFile
     * @param nameCol
     * @param lvl3Col
     * @throws IOException
     */
    private void process(String originalGSFile, String outputFile,
                         int nameCol,int urlCol,
                         int lvl3Col, int topN) throws IOException {

        CSVReader csvReader = GS1AnnotationUtil.getCSVReader(originalGSFile);
        CSVWriter csvWriter = GS1AnnotationUtil.getCSVWriter(outputFile);

        String[] nextRecord;

        int countLines = 0;
//        while ((nextRecord = csvReader.readNext()) != null) {
//            countLines++;
//        }
//        System.out.println(countLines);


        while ((nextRecord = csvReader.readNext()) != null) {
            countLines++;
            if (countLines == 1) {
                csvWriter.writeNext(new String[]{"Original_GS_name","url","lvl3_label",
                "name_similarity","new_name","label", "url","desc","id"});
                continue;
            }

            String name = nextRecord[nameCol];
            String url = nextRecord[urlCol];
            System.out.println(String.format("currently line %d, name=%s", countLines, name));
            List<String[]> similar = getSimilarProducts(name, topN);


            /*
            - format:
 *      ref_prod name, ref_prod label; similarity, similar prod name, url, desc, label, id
 *                                                    ... (repeat 10)
             */

            if (similar == null) {
                System.err.println(String.format("\t product %s has NULL name after normalisation:", name));
                continue;
            }

            for (String[] sim : similar) {
                String[] line = new String[9];
                line[0] = name;
                line[1]=url;
                line[2] = nextRecord[lvl3Col];
                line[3] = sim[0];
                line[4] = sim[1];
                line[5]="";
                line[6] = sim[2];
                line[7] = sim[3];
                line[8] = sim[4];
                csvWriter.writeNext(line);
            }

        }
        csvReader.close();
        csvWriter.close();
        System.out.println("Total lines="+countLines);
    }
}
