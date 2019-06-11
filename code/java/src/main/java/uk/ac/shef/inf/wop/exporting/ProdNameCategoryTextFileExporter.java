package uk.ac.shef.inf.wop.exporting;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;

import java.io.FileWriter;
import java.io.IOException;
/*
 * This class reads the index created by ProdNameCategoryExporter, to export product name and cat label into batches of txt files
 */
public class ProdNameCategoryTextFileExporter {
    private static final Logger LOG = Logger.getLogger(ProdNameCategoryTextFileExporter.class.getName());
    private long maxLinesPerFile=100000;
    //private long maxWordsPerFile=500;
    private int catFilecounter=0;
    private CSVWriter catFile;

    private void export(SolrClient prodcatIndex, int resultBatchSize, String outFolder) throws IOException {
        int start = 0;
        SolrQuery q = createQuery(resultBatchSize, start);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        FileWriter outputfile = new FileWriter(outFolder+"/c_"+catFilecounter+".csv",true);
        // create CSVWriter object filewriter object as parameter
        catFile = new CSVWriter(outputfile,',','"');

        int countCatFileLines=0;

        while (!stop) {
            try {
                res = prodcatIndex.query(q);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));

                for (SolrDocument d : res.getResults()) {
                    //process and export to the other solr index
                    int lines= exportRecord(d, catFile);
                    countCatFileLines += lines;
                }

                if (countCatFileLines>=maxLinesPerFile){
                    catFile.close();
                    catFilecounter++;
                    catFile=new CSVWriter(

                            new FileWriter(outFolder+"/c_"+catFilecounter+".csv",true));
                    countCatFileLines=0;
                }


                start = start + resultBatchSize;
            } catch (Exception e) {
                LOG.warn(String.format("\t\t unable to successfully index product triples starting from index %s. Due to error: %s",
                        start,
                        ExceptionUtils.getFullStackTrace(e)));

            }

            int curr = q.getStart() + q.getRows();
            if (curr < total)
                q.setStart(curr);
            else
                stop = true;
        }

        try {
            prodcatIndex.close();
        } catch (Exception e) {
            LOG.warn(String.format("\t\t unable to shut down servers due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    private int exportRecord(SolrDocument d,
                             CSVWriter catFile) {

        Object nameData=d.getFieldValue("name");
        Object catData=d.getFirstValue("category");

        if (nameData!=null && catData!=null){
            String name = cleanData(nameData.toString());
            String cat = cleanData(catData.toString());

            if (name.split("\\s+").length>1 && cat.split("\\s+").length>0){
                String[] values = {name, cat};

                catFile.writeNext(values);
                return 1;
            }
            else
                return 0;
        }

        return 0;
    }

    private String cleanData(String value){
        value= StringEscapeUtils.unescapeJava(value);
        value=value.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]"," ");
        value=value.replaceAll("\\s+"," ");
        value= StringUtils.stripAccents(value);
        return value.trim();
    }

    private SolrQuery createQuery(int resultBatchSize, int start){
        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setSort("random_1234", SolrQuery.ORDER.desc);
        query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    public static void main(String[] args) throws IOException {
        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodCatIndex = new EmbeddedSolrServer(prodNDContainer.getCore("prodcat"));

        ProdNameCategoryTextFileExporter exporter = new ProdNameCategoryTextFileExporter();
        //exporter.export(prodTripleIndex, Integer.valueOf(args[2]), prodNameDescIndex);
        exporter.export(prodCatIndex, 500,args[1]);
        prodCatIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }

}
