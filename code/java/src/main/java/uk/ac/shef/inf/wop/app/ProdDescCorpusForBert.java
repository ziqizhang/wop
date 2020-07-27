package uk.ac.shef.inf.wop.app;

import com.google.gson.Gson;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.client.solrj.util.ClientUtils;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import org.apache.solr.search.QueryUtils;
import uk.ac.shef.inf.wop.exporting.ProdDescTextFileExporter_Lucene;

import java.io.*;

import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * given a product name, search in a product desc index to find similar products, and create its description by concatenating
 * and selecting these product descriptions
 *
 * These are written to a file on a line-by-line basis.
 *
 */
public class ProdDescCorpusForBert {

    private static final Logger LOG = Logger.getLogger(ProdDescCorpusForBert.class.getName());
    private static final Map<String, Integer> repeatedRecords=new HashMap<>();

    public static void main(String[] args) throws IOException {
        /*String test="{\"ID\": \"549\", \"Name\": \"Sterling Silver Angel Charm\", \"Description\": \"This little angel charm is just heavenly\", \"CategoryText\": \"All Products\", \"URL\": \"http://www.thecharmworks.com/product/CW-UA/Sterling-Silver-Angel-Charm.html\", \"lvl1\": \"64000000_Personal Accessories\", \"lvl2\": \"64010000_Personal Accessories\", \"lvl3\": \"64010100_Jewellery\"}";
        Gson googleJson = new Gson();*/

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("prodcatdesc"));

        int maxResults = Integer.valueOf(args[1]);//max number of products to select for composing desc
        String inFile = args[2];
        String nameCol=args[3];
        String outFile = args[4];

        exportDesc(inFile, outFile, nameCol, maxResults, prodNameDescIndex);

        prodNameDescIndex.close();
        System.exit(0);
    }

    private static void exportDesc(String inFile, String outFile,
                                   String nameCol, int maxResults,
                                   SolrClient prodNameDescIndex) {
        try {

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), StandardCharsets.UTF_8));
            Gson googleJson = new Gson();
            int countRecords = 0;
            OutputStreamWriter writer =
                    new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8);

            String line;
            while ((line = br.readLine()) != null) {
                Map<String, String> rowValues=googleJson.fromJson(line, Map.class);

                //List row_values = (ArrayList) row;
                String name=rowValues.get(nameCol);
                if (name==null){
                    System.err.println("Line "+countRecords+" has no name, skip");
                    continue;
                }
                if (name.endsWith("-"))
                    name=name.substring(0, name.length()-1).trim();
                String newDesc = expand(name, prodNameDescIndex, maxResults);

                //nextRecord[WOP_DESC_COL] = newDesc;
                writer.write(newDesc + "\n");
                System.out.println(countRecords);
                countRecords++;

                if (countRecords%100==0)
                    System.out.println("\t done "+countRecords);
            }
            writer.close();

            List<String> ids = new ArrayList<>(repeatedRecords.keySet());
            ids.sort((s, t1) -> repeatedRecords.get(t1).compareTo(repeatedRecords.get(s)));
            System.out.println("Total records="+countRecords);
            System.out.println("Total unique selected="+repeatedRecords.size());
            System.out.println("Repetitions:");
            for (Map.Entry<String, Integer> e: repeatedRecords.entrySet()){
                System.out.println(e.getValue()+"\t"+e.getKey());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }




    }

    /**
     * @param productName
     * @param solrIndex
     */
    public static String expand(String productName, SolrClient solrIndex, int maxResults) {
        SolrQuery q = createQuery(100, productName);
        QueryResponse res;
        long total = 0;
        List<String> selected = new ArrayList<>();
        StringBuilder desc=new StringBuilder();
        try {
            res = solrIndex.query(q);
            if (res != null)
                total = res.getResults().getNumFound();
            //update results
            LOG.info(String.format("\t\tprocessing from %s, %d results ...",
                    productName, total));

            int countResults = 0;
            for (SolrDocument d : res.getResults()) {
                String id = getStringValue(d, "id");
                int freq=0;
                if (repeatedRecords.containsKey(id)){
                    freq = repeatedRecords.get(id);
                }
                freq+=1;
                repeatedRecords.put(id, freq);


                String name = getStringValue(d, "name");
                if (name != null) {
                    name = removeSpaces(name);
                    if (name.toLowerCase().equalsIgnoreCase(productName.toLowerCase()))
                        continue;
                }
                String vdesc = getStringValue(d, "desc");
                vdesc = ProdDescTextFileExporter_Lucene.cleanData(vdesc);
                String[] tokens = vdesc.split("\\s+");
                if (vdesc.length() > 20 && tokens.length >= ProdDescTextFileExporter_Lucene.MIN_DESC_WORDS) {
                    if (tokens.length>ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS)
                        vdesc= StringUtils.join(tokens, " ",0,ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS);

                    selected.add(vdesc);
                    countResults++;
                }

                if (countResults >= maxResults) {
                    break;
                }
            }


        } catch (Exception e) {
            LOG.warn(String.format("\t\t\t error encountered, skipped due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));

        }

        for (String d : selected)
            desc.append(d).append("\n");
        return desc.toString().replaceAll("\\s+"," ").trim();
    }

    private static SolrQuery createQuery(int resultBatchSize, String name) {
        SolrQuery query = new SolrQuery();
        query.setQuery("desc:" + ClientUtils.escapeQueryChars(name));
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

    public class GenericJson<T> {

        public List<T> values;

        public List<T> getValues() {
            return values;
        }
    }
}
