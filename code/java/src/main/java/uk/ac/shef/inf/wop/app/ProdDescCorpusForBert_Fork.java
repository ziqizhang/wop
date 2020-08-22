package uk.ac.shef.inf.wop.app;

import com.google.gson.Gson;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.core.CoreContainer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ForkJoinPool;

public class ProdDescCorpusForBert_Fork {
    private static final Logger LOG = Logger.getLogger(ProdDescCorpusForBert_Fork.class.getName());

    public static void main(String[] args) throws IOException {

        ProdDescCorpusForBert_ConcurrentSet results = new ProdDescCorpusForBert_ConcurrentSet();

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("prodcatdesc"));
        System.out.println(new Date()+"\tSolr server ready");

        //SolrClient predicatesCoreClient= new EmbeddedSolrServer(solrContainer.getCore("predicates"));
        String[] maxResults = args[1].split(",");//max number of products to select for composing desc

        String inFile = args[2];
        String nameCol=args[3];
        String outFolder = args[4];

        int threads=Integer.valueOf(args[6]);

        double sample=1.0;
        if (args.length>7)
            sample=Double.valueOf(args[7]);

        System.out.println(new Date()+"\tReading data records...");
        List<String> tasks = readTasks(inFile, args[5], nameCol);
        int maxPerThread = tasks.size()/threads +1;


        ProdDescCorpusForBert_Join pt = new
                ProdDescCorpusForBert_Join(0,tasks,
                outFolder, maxResults, prodNameDescIndex, args[5], sample, results, maxPerThread);

        try {

            System.out.println(new Date()+"\tStarting all processes");
            ForkJoinPool forkJoinPool = new ForkJoinPool();
            int total = forkJoinPool.invoke(pt);

            LOG.info(String.format("Completed, total entities=%s", total));

        } catch (Exception ioe) {
            StringBuilder sb = new StringBuilder("Failed to build features!");
            sb.append("\n").append(ExceptionUtils.getFullStackTrace(ioe));
            LOG.error(sb.toString());
        }


        prodNameDescIndex.close();
        System.exit(0);
    }

    private static List<String> readTasks(String inFile, String dataset,
                                          String nameCol) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(inFile), StandardCharsets.UTF_8));
        Gson googleJson = new Gson();
        String line;
        int countRecords = 0;
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
        }

        return allRecords;
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
