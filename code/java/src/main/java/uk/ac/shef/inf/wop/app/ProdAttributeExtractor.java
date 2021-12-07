package uk.ac.shef.inf.wop.app;

import dragon.nlp.tool.lemmatiser.EngLemmatiser;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.wop.Lemmatiser;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.*;

/**
 * given a product name, search in a product desc index to find 100 similar products, then choose the top 20 different
 * product names. Next, for each word in the product name, count the freqency found in the
 * reference product names. Finally, choose the N highest frequent words
 *
 *
 */

public class ProdAttributeExtractor {

    private static final Logger LOG = Logger.getLogger(ProdAttributeExtractor.class.getName());

    public static void main(String[] args) throws IOException {

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("proddesc"));

        int refProds2Retrieve = Integer.valueOf(args[1]); //max number of sentences in composed desc
        int refNames2Take = Integer.valueOf(args[2]);//max number of products to select for composing desc
        String inFile = args[3];
        String outFile = args[4];
        String dataset = args[5];
        int att2select=Integer.valueOf(args[6]);
        String lemmatiserRes=args[7];

        Lemmatiser lem = new Lemmatiser(new EngLemmatiser(
                lemmatiserRes, false, false
        ));

        if (dataset.equalsIgnoreCase("wop")) {
            extract(inFile, outFile, refProds2Retrieve, refNames2Take, att2select, prodNameDescIndex, lem);
        } else {
            //todo: for Rakuten data
        }
        prodNameDescIndex.close();
        System.exit(0);
    }

    private static void extract(String inFile, String outFile,
                                int refProds2Retrieve, int refProdNames2Take, int att2select,
                                SolrClient prodNameDescIndex,
                                Lemmatiser lem) {
        try {
            List<String> lines = FileUtils.readLines(new File(inFile), "utf-8");
            // Reading Records One by One in a String array
            String[] nextRecord;
            int countline = 0;
            OutputStreamWriter writer =
                    new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8);
            for (String name : lines) {
                if (name.endsWith("-"))
                    name=name.substring(0, name.length()-1).trim();
                String attributes =
                        extract(name, prodNameDescIndex, refProds2Retrieve, refProdNames2Take, att2select,lem);

                //nextRecord[WOP_DESC_COL] = newDesc;
                writer.write(attributes + "\n");
                System.out.println(countline);
                countline++;
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    /**
     * @param productName
     * @param solrIndex
     * @param refProds2Retrieve   of description to generate. Currently in terms of sentences.
     */
    public static String extract(String productName, SolrClient solrIndex,
                                 int refProds2Retrieve, int refProdNames2Take, int att2Select,
                                 Lemmatiser lem) {
        //prepare product name
        productName = normalise(productName);
        Map<String, Integer> freq = new HashMap<>();
        Map<String, Integer> index=new HashMap<>();
        int idx=0;
        for (String t: productName.split("\\s+")){
            t=lem.normalize(t, "NN");
            idx++;
            if (t.length()<3)
                continue;
            freq.put(t,0);
            index.put(t, idx);
        }

        //perform query to get similar product names
        SolrQuery q = createQuery(refProds2Retrieve, productName);
        QueryResponse res;
        long total = 0;
        Set<String> selected = new HashSet<>();
        try {
            res = solrIndex.query(q);
            if (res != null)
                total = res.getResults().getNumFound();
            //update results
            LOG.info(String.format("\t\tprocessing from %s, %d results ...",
                    productName, total));

            int countRefProds = 0, countResults = 0;
            for (SolrDocument d : res.getResults()) {
                String name = getStringValue(d, "name");
                if (name != null) {
                    name = normalise(name);
                    if (name.equalsIgnoreCase(productName.toLowerCase()))
                        continue;
                }

                selected.add(name);

                if (selected.size() >= refProdNames2Take) {
                    break;
                }
            }


        } catch (Exception e) {
            LOG.warn(String.format("\t\t\t error encountered, skipped due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));

        }

        //collect stats and rank
        StringBuilder attributes = new StringBuilder();
        for (String name : selected){
            String[] toks = name.split("\\s+");
            for (String t: toks){
                t=lem.normalize(t, "NN");
                if (t.length()<3)
                    continue;
                if (freq.containsKey(t)){
                    freq.put(t, freq.get(t)+1);
                }
            }
        }

        List<String> words=new ArrayList<>(freq.keySet());
        words.sort((t1, t2) -> {
            Integer rank1 = freq.get(t1);
            Integer rank2 = freq.get(t2);
            if (rank1.equals(rank2)) {
                Integer idx1 = index.get(t1);
                Integer idx2 = index.get(t2);
                return idx2.compareTo(idx1);
            } else
                return rank2.compareTo(rank1);

        });

        List<String> attSelected= new ArrayList<>();
        for(int i=0; i<att2Select && i <words.size(); i++){
            attSelected.add(words.get(i));
        }
        attSelected.sort((t1, t2) -> {
            Integer idx1 = index.get(t1);
            Integer idx2 = index.get(t2);
            return idx1.compareTo(idx2);
        });
        for (String a : attSelected)
            attributes.append(a).append(" ");
        return attributes.toString().trim();
    }

    private static SolrQuery createQuery(int resultBatchSize, String name) {
        SolrQuery query = new SolrQuery();
        query.setQuery("text:" + name);
        //query.setSort("random_1234", SolrQuery.ORDER.asc);
        query.setStart(0);
        query.setRows(resultBatchSize);

        return query;
    }

    private static String getStringValue(SolrDocument doc, String field) {
        Object v = doc.getFieldValue(field);
        if (v != null)
            return v.toString();
        else
            return null;
    }

    private static String normalise(String v) {
        return v.replaceAll("\\s+", " ").trim().toLowerCase();
    }
}
