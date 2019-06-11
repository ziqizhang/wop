package uk.ac.shef.inf.wop.exporting;

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
import java.io.PrintWriter;

/**
 * This class reads the index created by ProdDescExporter, to export product descriptions into batches of txt files
 */
public class ProdDescTextFileExporter {

    private static final Logger LOG = Logger.getLogger(ProdDescTextFileExporter.class.getName());
    private long maxWordsPerFile=5000000000L;
    //private long maxWordsPerFile=500;
    private int nameFileCounter=0, descFileCounter=0;
    private PrintWriter nameFile;
    private PrintWriter descFile;

    public void export(SolrClient prodNameDescIndex, int resultBatchSize, String nameOutFolder, String descOutFolder) throws IOException {
        int start = 0;
        SolrQuery q = createQuery(resultBatchSize, start);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        nameFile=new PrintWriter(new FileWriter(nameOutFolder+"/n_"+nameFileCounter,true));
        descFile=new PrintWriter(new FileWriter(descOutFolder+"/d_"+descFileCounter,true));

        long countNameFileWords=0, countDescFileWords=0;

        while (!stop) {
            try {
                res = prodNameDescIndex.query(q);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));

                for (SolrDocument d : res.getResults()) {
                    //process and export to the other solr index
                    long[] words= exportRecord(d, nameFile, descFile);
                    countNameFileWords += words[0];
                    countDescFileWords += words[1];
                }

                if (countNameFileWords>=maxWordsPerFile){
                    nameFile.close();
                    nameFileCounter++;
                    nameFile=new PrintWriter(new FileWriter(nameOutFolder+"/n_"+nameFileCounter,true));
                    countNameFileWords=0;
                }
                if(countDescFileWords>=maxWordsPerFile){
                    descFile.close();
                    descFileCounter++;
                    descFile=new PrintWriter(new FileWriter(descOutFolder+"/d_"+descFileCounter,true));
                    countDescFileWords=0;
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
            prodNameDescIndex.close();
            nameFile.close();
            descFile.close();
        } catch (Exception e) {
            LOG.warn(String.format("\t\t unable to shut down servers due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    private long[] exportRecord(SolrDocument d,
                                PrintWriter nameFile, PrintWriter descFile) {

        Object nameData=d.getFieldValue("name");
        Object descData=d.getFirstValue("text");
        long[] res=new long[2];

        if (nameData!=null){
            String name = cleanData(nameData.toString());
            long tokens=name.split("\\s+").length;
            if(name.length()>10 && tokens>2) {
                nameFile.println(name);
                res[0] = tokens;
            }
        }
        if (descData!=null){
            String desc = cleanData(descData.toString());
            long tokens=desc.split("\\s+").length;
            if(desc.length()>20 && tokens>5) {
                descFile.println(desc);
                res[1] = tokens;
            }
        }

        return res;
    }

    private String cleanData(String value){
        value= StringEscapeUtils.unescapeJava(value);
        value=value.replaceAll("\\s+"," ");
        value= StringUtils.stripAccents(value);
        return value.trim();
    }

    private SolrQuery createQuery(int resultBatchSize, int start){
        SolrQuery query = new SolrQuery();
        query.setQuery("text:* OR name:*");
        query.setSort("random_1234", SolrQuery.ORDER.desc);
        query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    public static void main(String[] args) throws IOException {
        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("proddesc"));

        ProdDescTextFileExporter exporter = new ProdDescTextFileExporter();
        //exporter.export(prodTripleIndex, Integer.valueOf(args[2]), prodNameDescIndex);
        exporter.export(prodNameDescIndex, 500,args[1],args[2]);
        System.exit(0);
        LOG.info("COMPLETE!");

    }
}
