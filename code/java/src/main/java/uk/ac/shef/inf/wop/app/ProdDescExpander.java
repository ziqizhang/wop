package uk.ac.shef.inf.wop.app;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;

import java.io.*;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * given a product name, search in a product desc index to find similar products, and create its description by concatenating
 * and selecting these product descriptions
 *
 * These are written to a file on a line-by-line basis.
 *
 * To use that file, use 'lanmodel/ir_desc_cat_expander.py to read this output file and merge it with gs dataset
 */
public class ProdDescExpander {

    private static final Logger LOG = Logger.getLogger(ProdDescExpander.class.getName());

    public static void main(String[] args) throws IOException {

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("prodcatdesc"));

        int descSents = Integer.valueOf(args[1]); //max number of sentences in composed desc
        int maxResults = Integer.valueOf(args[2]);//max number of products to select for composing desc
        String inFile = args[3];
        String outFile = args[4];
        String dataset = args[5];

        if (dataset.equalsIgnoreCase("wop")) {
            exportDesc(inFile, outFile, descSents, maxResults, prodNameDescIndex);
        } else {
            //todo: for Rakuten data
        }
        prodNameDescIndex.close();
        System.exit(0);
    }

    private static void exportDesc(String inFile, String outFile,
                                   int descSents, int maxResults,
                                   SolrClient prodNameDescIndex) {
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
                String newDesc = expand(name, prodNameDescIndex, descSents, maxResults);

                //nextRecord[WOP_DESC_COL] = newDesc;
                writer.write(newDesc + "\n");
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
     * @param descSents   of description to generate. Currently in terms of sentences.
     */
    public static String expand(String productName, SolrClient solrIndex, int descSents, int maxResults) {
        SolrQuery q = createQuery(100, productName);
        QueryResponse res;
        long total = 0;
        StringBuilder desc = new StringBuilder();
        List<String> selected = new ArrayList<>();
        try {
            res = solrIndex.query(q);
            if (res != null)
                total = res.getResults().getNumFound();
            //update results
            LOG.info(String.format("\t\tprocessing from %s, %d results ...",
                    productName, total));

            int countDescSents = 0, countResults = 0;
            for (SolrDocument d : res.getResults()) {
                String name = getStringValue(d, "name");
                if (name != null) {
                    name = removeSpaces(name);
                    if (name.toLowerCase().equalsIgnoreCase(productName.toLowerCase()))
                        continue;
                }
                String vdesc = getStringValue(d, "text");
                if (vdesc == null || vdesc.length() < 20)
                    continue;

                if (selected.contains(vdesc))
                    continue;

                selected.add(vdesc);

                countDescSents += vdesc.split("\\.").length;
                countResults++;

                if (countDescSents >= descSents || countResults >= maxResults) {
                    break;
                }
            }


        } catch (Exception e) {
            LOG.warn(String.format("\t\t\t error encountered, skipped due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));

        }

        for (String d : selected)
            desc.append(d).append(" ");
        return desc.toString().replaceAll("\\s+"," ").trim();
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

    private static String removeSpaces(String v) {
        return v.replaceAll("\\s+", " ").trim();
    }
}
