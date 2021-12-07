/**
 * This is the correct class to use to create 'simco' corpus for bert lmft
 *
 /home/zz/Work/data/wdc/WDCNov2017/prodcatdesc
 1,5,10
 /home/zz/Work/data/IceCAT/icecat_data_train.json
 Name
 /home/zz/Work/data/IceCAT
 icecat
 1


 /home/zz/Work/data/wdc/WDCNov2017/prodcatdesc
 1,5,10
 /home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS/wdc_gs_train.json
 Name
 /home/zz/Cloud/GDrive/ziqizhang/project/mwpd/prodcls/data/WDC_CatGS
 wdc
 4

 /home/zz/Work/data/wdc/WDCNov2017/prodcatdesc
 1,5,10
 /home/zz/Work/data/IceCAT/icecat_data_train.json
 Name
 /home/zz/Work/data/IceCAT
 icecat
 4
 */

package uk.ac.shef.inf.wop.app;

import com.google.gson.Gson;
import org.apache.commons.collections4.ListUtils;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ProdDescCorpusForBert_Thread implements Runnable {

    private static final Logger LOG = Logger.getLogger(ProdDescCorpusForBert_Thread.class.getName());

    public static void main(String[] args) throws IOException {
        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("prodcatdesc"));
        ProdDescCorpusForBert_ConcurrentSet results = new ProdDescCorpusForBert_ConcurrentSet();

        String[] maxResults = args[1].split(",");//max number of products to select for composing desc

        String inFile = args[2];
        String nameCol=args[3];
        String outFolder = args[4];
        String dataset=args[5];

        int threads=Integer.valueOf(args[6]);
        System.out.println("Reading data...");
        List<List<String>> tasks = readTasks(inFile, dataset, nameCol, threads);

        double sample=1.0;
        if (args.length>7)
            sample=Double.valueOf(args[7]);

        ExecutorService executor = Executors.newFixedThreadPool(threads);
        for (int i = 0; i < threads; i++) {
            List<String> task = tasks.get(i);
            Runnable exporter = new ProdDescCorpusForBert_Thread(i,
                    task,
                    outFolder,
                    maxResults,
                    prodNameDescIndex,
                    dataset,
                    sample,
                    results);
            executor.execute(exporter);
        }
        executor.shutdown();
        while (!executor.isTerminated()) {
        }

        prodNameDescIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }

    private List<String> tasks;
    private int threadID;
    private String outFolder;
    private String[] maxResults;
    private SolrClient prodNameDescIndex;
    private String dataset;
    private double sample;
    private ProdDescCorpusForBert_ConcurrentSet allSelected;

    public ProdDescCorpusForBert_Thread(int id,
                                      List<String> tasks,
                                      String outFolder,
                                      String maxResults[],
                                      SolrClient prodNameDescIndex,
                                      String dataset,
                                      double sample,
                                      ProdDescCorpusForBert_ConcurrentSet allSelected) {
        this.threadID = id;
        this.tasks = tasks;
        this.outFolder = outFolder;
        this.maxResults = maxResults;
        this.prodNameDescIndex = prodNameDescIndex;
        this.sample = sample;
        this.dataset = dataset;
        this.allSelected = allSelected;
    }

    private static List<List<String>> readTasks(String inFile, String dataset,
                                          String nameCol,
                                                int thread) throws IOException {

        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), StandardCharsets.UTF_8));
        Gson googleJson = new Gson();
        String line;
        int countRecords = 0;
        int maxLines=Integer.MAX_VALUE;
        List<String> allRecords = new ArrayList<>();
        while ((line = br.readLine()) != null) {
               /* if (countRecords>=114)
                    System.out.println();*/
            Map rowValues;
            if (dataset.equalsIgnoreCase("mwpd"))
                rowValues = readMPWDLine(line, googleJson);
            else if (dataset.equalsIgnoreCase("wdc"))
                rowValues = readWDCLine(line, googleJson);
            else if (dataset.equalsIgnoreCase("icecat"))
                rowValues = readIceCatLine(line, googleJson);
            else
                rowValues = readRakutenLine(line);

            //List row_values = (ArrayList) row;
            String name = (String) rowValues.get(nameCol);
            if (name == null || name.length() == 0) {
                System.err.println("Line " + countRecords + " has no name, skip");
                continue;
            }
            if (name.endsWith("-"))
                name = name.substring(0, name.length() - 1).trim();
            if (name.length() == 0) {
                continue;
            }

            if (!allRecords.contains(name)) {
                allRecords.add(name);
                countRecords++;
            }
            if (countRecords%1000 ==0)
                System.out.println(new Date()+" \t read "+countRecords+" data records");

            if (countRecords>=maxLines)
                break;
        }

        int size = allRecords.size()/thread +1;
        return ListUtils.partition(allRecords, size);
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

    @Override
    public void run() {
        int total_selected=0;
        try {
            Set<Integer> sample_lines = new HashSet<>();
            if (sample < 1.0) {
                System.out.println("<<THREAD " + threadID + ">> Processing only sample size=" + sample + ", or " + sample_lines.size() + " records");
                List<Integer> all_lines = new ArrayList<>();

                int countAll = 0;
                for (String t : tasks) {
                    all_lines.add(countAll);
                    countAll++;
                }
                int toSelect = (int) (sample * countAll);
                Collections.shuffle(all_lines);
                sample_lines = new HashSet<>(all_lines.subList(0, toSelect));
            }

            //Gson googleJson = new Gson();
            int countRecords = 0;

            System.out.println(new Date()+"\t<<THREAD " + threadID + ">> Processing data size=" + tasks.size());
            Map<Integer, OutputStreamWriter> writers = new HashMap<>();
            for (String maxR : maxResults) {
                String outDir = outFolder + "/" + maxR;
                new File(outDir).mkdirs();

                String outFile = outDir + "/thread" + threadID + "_" + dataset + ".txt";
                OutputStreamWriter writer =
                        new OutputStreamWriter(new FileOutputStream(outFile), StandardCharsets.UTF_8);
                writers.put(Integer.valueOf(maxR), writer);
            }

            for (String name : tasks) {
                if (sample_lines.size() > 0 && !sample_lines.contains(countRecords)) {
                    countRecords++;
                    continue;
                }


                //System.out.println(new Date()+"\tRead "+ name);
                total_selected += expand(name, writers);

                //System.out.println(new Date()+"\t\tProcessed "+ name);
                //nextRecord[WOP_DESC_COL] = newDesc;
                countRecords++;

                if (countRecords % 100 == 0)
                    System.out.println("<<THREAD " + threadID + ">> " + new Date() + " \t done " + countRecords);
            }

            for (OutputStreamWriter w : writers.values())
                w.close();

            System.out.println("<<THREAD " + threadID + ">> Total records=" + countRecords);

        }catch (Exception e){
            e.printStackTrace();
        }
    }

    /**
     * @param productName
     */
    private int expand(String productName, Map<Integer, OutputStreamWriter> maxR_and_writers) {
        Set<Integer> maxRs = new HashSet<>(maxR_and_writers.keySet());
        //Set<Integer> maxRs = new HashSet<>();

        SolrQuery q = createQuery(200, productName);
        QueryResponse res;
        long total = 0;
        List<String> selected = new ArrayList<>();

        try {
            res = prodNameDescIndex.query(q);
            if (res != null)
                total = res.getResults().getNumFound();
            //update results
            /*LOG.info(String.format("\t\tprocessing from %s, %d results ...",
                    productName, total));*/

            int countResults = 0;
            for (SolrDocument d : res.getResults()) {
                String docid = getStringValue(d, "id");
                if (maxRs.size() == 0)
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
                if (vdesc.length() > 20 && tokens.length >= ProdDescTextFileExporter_Lucene.MIN_DESC_WORDS
                        /*&& vdesc.length() < ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS * 10*/) {
                    if (tokens.length > ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS)
                        vdesc = StringUtils.join(tokens, " ", 0, ProdDescTextFileExporter_Lucene.MAX_DESC_WORDS);

                    countResults++;

                    //for each doc only select ones
                    /*if (!allSelected.contains(docid)) {
                        selected.add(vdesc);
                        allSelected.add(docid);
                    }*/

                    //allowing selecting multiple times of the same doc
                    selected.add(vdesc);
                    allSelected.add(docid);
                }

                //check if we should dump for any 'max Results' writer
                int finish_maxR = -1;
                for (int mr : maxRs) {
                    if (countResults >= mr) {
                        finish_maxR = mr;
                        OutputStreamWriter writer = maxR_and_writers.get(mr);
                        for (String description : selected)
                            writer.write(description + "\n");
                    }
                }
                maxRs.remove(finish_maxR);
            }


        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(String.format("\t\t\t error encountered, skipped due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
        return selected.size();

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
}
