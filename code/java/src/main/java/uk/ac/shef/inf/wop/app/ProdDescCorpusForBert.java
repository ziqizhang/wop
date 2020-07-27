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
    private static final Set<String> allSelected =new HashSet<>();

    public static void main(String[] args) throws IOException {
        /*String test="{\"ID\": \"549\", \"Name\": \"Sterling Silver Angel Charm\", \"Description\": \"This little angel charm is just heavenly\", \"CategoryText\": \"All Products\", \"URL\": \"http://www.thecharmworks.com/product/CW-UA/Sterling-Silver-Angel-Charm.html\", \"lvl1\": \"64000000_Personal Accessories\", \"lvl2\": \"64010000_Personal Accessories\", \"lvl3\": \"64010100_Jewellery\"}";
        Gson googleJson = new Gson();*/

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("prodcatdesc"));

        String[] maxResults = args[1].split(",");//max number of products to select for composing desc

        String inFile = args[2];
        String nameCol=args[3];
        String outFolder = args[4];

        exportDesc(inFile, outFolder, nameCol, maxResults, prodNameDescIndex, args[5]);

        prodNameDescIndex.close();
        System.exit(0);
    }

    private static void exportDesc(String inFile, String outFolder,
                                   String nameCol, String[] maxResults,
                                   SolrClient prodNameDescIndex,
                                   String dataset) {
        try {

            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), StandardCharsets.UTF_8));
            Gson googleJson = new Gson();
            int countRecords = 0;

            Map<Integer, OutputStreamWriter> writers= new HashMap<>();
            for (String maxR: maxResults){
                String outFile = maxR+"_"+new File(inFile).getName()+".txt";
                OutputStreamWriter writer =
                        new OutputStreamWriter(new FileOutputStream(outFolder+"/"+outFile), StandardCharsets.UTF_8);
                writers.put(Integer.valueOf(maxR), writer);
            }


            String line;
            while ((line = br.readLine()) != null) {
               /* if (countRecords>=114)
                    System.out.println();*/
                Map rowValues;
                if (dataset.equalsIgnoreCase("mwpd"))
                    rowValues=readMPWDLine(line, googleJson);
                else if (dataset.equalsIgnoreCase("wdc"))
                    rowValues=readWDCLine(line, googleJson);
                else if (dataset.equalsIgnoreCase("icecat"))
                    rowValues=readIceCatLine(line, googleJson);
                else
                    rowValues=readRakutenLine(line);

                //List row_values = (ArrayList) row;
                String name=(String)rowValues.get(nameCol);
                if (name==null||name.length()==0){
                    System.err.println("Line "+countRecords+" has no name, skip");
                    continue;
                }
                if (name.endsWith("-"))
                    name=name.substring(0, name.length()-1).trim();
                if (name.length()==0) {
                    countRecords++;
                    continue;
                }

                expand(name, prodNameDescIndex, writers);

                //nextRecord[WOP_DESC_COL] = newDesc;
                countRecords++;

                if (countRecords%100==0)
                    System.out.println(new Date()+"\t done "+countRecords);
            }

            for (OutputStreamWriter w : writers.values())
                w.close();

            System.out.println("Total records="+countRecords);
            System.out.println("Total unique allSelected="+ allSelected.size());

        } catch (IOException e) {
            e.printStackTrace();
        }




    }

    /**
     * @param productName
     * @param solrIndex
     */
    public static void expand(String productName, SolrClient solrIndex, Map<Integer, OutputStreamWriter> maxR_and_writers) {
        Set<Integer> maxRs = new HashSet<>(maxR_and_writers.keySet());
        SolrQuery q = createQuery(200, productName);
        QueryResponse res;
        long total = 0;
        Set<String> selected=new HashSet<>();

        try {
            res = solrIndex.query(q);
            if (res != null)
                total = res.getResults().getNumFound();
            //update results
            /*LOG.info(String.format("\t\tprocessing from %s, %d results ...",
                    productName, total));*/

            int countResults = 0;
            for (SolrDocument d : res.getResults()) {
                String docid=getStringValue(d, "id");
                if (maxRs.size()==0)
                    break;

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

                    countResults++;

                    if (!allSelected.contains(docid)) {
                        selected.add(vdesc);
                        allSelected.add(docid);
                    }
                }

                //check if we should dump for any 'max Results' writer
                int finish_maxR =-1;
                for (int mr : maxRs){
                    if (countResults >=mr){
                        finish_maxR=mr;
                        OutputStreamWriter writer = maxR_and_writers.get(mr);
                        for (String description: selected)
                            writer.write(description+"\n");
                    }
                }
                maxRs.remove(finish_maxR);
            }


        } catch (Exception e) {
            LOG.warn(String.format("\t\t\t error encountered, skipped due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }


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

    private static Map readMPWDLine(String line, Gson gson){
        return gson.fromJson(line, Map.class);
    }

    private static Map readWDCLine(String line, Gson gson){
        Map values = gson.fromJson(line, Map.class);

        Map result=new HashMap();
        List l = null;
        if (values.containsKey("schema.org_properties")){
            l=(List)values.get("schema.org_properties");
            for (Object o : l){
                Map m = (Map)o;
                if (m.containsKey("/title")) {
                    result.put("Name", trim(String.valueOf(m.get("/title"))));
                    break;
                }
                if (m.containsKey("/name")) {
                    result.put("Name", trim(String.valueOf(m.get("/name"))));
                    break;
                }
            }

        }

        if (values.containsKey("parent_schema.org_properties")&&(!result.containsKey("Name")
                ||String.valueOf(result.get("Name")).length()==0)) {
            l = (List) values.get("parent_schema.org_properties");
            for (Object o : l){
                Map m = (Map)o;
                if (m.containsKey("/title")) {
                    result.put("Name", trim(String.valueOf(m.get("/title"))));
                    break;
                }
                if (m.containsKey("/name")) {
                    result.put("Name", trim(String.valueOf(m.get("/name"))));
                    break;
                }
            }
        }

        return result;
    }

    private static Map readRakutenLine(String line){
        String[] values=line.split("\\t");
        Map res  =new HashMap();
        res.put("Name",values[0]);
        return res;
    }

    private static Map readIceCatLine(String line,Gson gson){
        //Title
        Map res = gson.fromJson(line, Map.class);
        if (res.containsKey("Title"))
            res.put("Name", res.get("Title"));
        return res;
    }

    private static String trim(String v){
        if (v.startsWith("["))
            v=v.substring(1);
        if (v.endsWith("]"))
            v=v.substring(0, v.length()-1);
        return v.trim();
    }
}
