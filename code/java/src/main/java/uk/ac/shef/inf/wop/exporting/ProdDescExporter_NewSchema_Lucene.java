package uk.ac.shef.inf.wop.exporting;

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.core.CoreContainer;
import org.apache.solr.search.SolrIndexSearcher;

import java.io.IOException;
import java.net.URL;

/**
 * processes the original desc index, parse 'id' field to extract host and save it as a separate field
 */
public class ProdDescExporter_NewSchema_Lucene {
    private static final Logger LOG = Logger.getLogger(ProdDescExporter_NewSchema_Lucene.class.getName());

    private int id;
    private int start;
    private int end;
    private IndexReader luceneIndexReader;
    private SolrClient prodNameDescIndex;
    private int resultBatchSize;

    public ProdDescExporter_NewSchema_Lucene(int id, int start, int end,
                                             IndexReader prodNameDescIndex_old,
                                             int resultBatchSize, SolrClient prodNameDescIndex_new) {
        this.id = id;
        this.start = start;
        this.end = end;
        this.luceneIndexReader = prodNameDescIndex_old;
        this.resultBatchSize = resultBatchSize;
        this.prodNameDescIndex = prodNameDescIndex_new;
    }

    public void export() {
        long total = 0;

        LOG.info(String.format("\tthread %d: Started, begin=%d end=%d...",
                id, start, end));

        try {

            long countAdded = 0;
            //update results
            LOG.info(String.format("\t\ttotal=%d", luceneIndexReader.maxDoc()));
            for (int i = start; i < luceneIndexReader.maxDoc() && i < end; i++) {

                Document doc = luceneIndexReader.document(i);
                boolean added = exportRecord(doc, prodNameDescIndex);
                if (added)
                    countAdded++;
                if (i % resultBatchSize == 0) {
                    LOG.info(String.format("\t\tthread %d: total results of %d, currently processing %d /started at %d to %d...",
                            id, total, i, start, end));
                    prodNameDescIndex.commit();
                }

                // do something with docId here...

            }

            prodNameDescIndex.commit();
        } catch (Exception ioe) {
            LOG.warn(String.format("\t\t thread %d unable to create output files, io exception: %s",
                    id, ExceptionUtils.getFullStackTrace(ioe)));
        }
    }

    private boolean exportRecord(Document d,
                                 SolrClient newIndex) throws IOException, SolrServerException {

        String id = d.get("id");
        String nameData = d.get("name");
        String descData = d.get("text");
        String host="";

        String url=id.split("\\|")[1].trim();
        try{
            URL u = new URL(url);
            host=u.getHost();
        }catch (Exception e){
            System.err.println("\t\t encountered invalid url in id:"+id);
            return false;
        }

        if (nameData != null) {
            String name = cleanData(nameData);
            long tokens = name.split("\\s+").length;
            if (name.length() > 10 && tokens > 2) {
                nameData = name;
            } else
                return false;
        }
        if (descData != null) {
            String desc = cleanData(descData);
            long tokens = desc.split("\\s+").length;
            if (desc.length() > 20 && tokens > 5) {
                descData = desc;
            } else
                return false;
        }

        SolrInputDocument newD = new SolrInputDocument();
        try {
            newD.addField("id", id);
            newD.addField("host", host);
            newD.addField("name", nameData);
            newD.addField("text", descData);
            newIndex.add(newD);
        }catch (Exception e){
            LOG.error("\t\tinvalid host:"+host);
        }
        return true;
    }

    private String cleanData(String value) {
        value = StringEscapeUtils.unescapeJava(value);
        value = value.replaceAll("\\s+", " ");
        value = StringUtils.stripAccents(value);
        return value.trim();
    }


    public static void main(String[] args) throws IOException {

        //74488335
        int jobStart = Integer.valueOf(args[2]);
        int jobs = Integer.valueOf(args[3]);

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrIndexSearcher solrIndexSearcher = prodNDContainer.getCore("proddesc").getSearcher().get();
        IndexReader prodNameDescIndex = solrIndexSearcher.getIndexReader();

        CoreContainer prodNCContainer = new CoreContainer(args[1]);
        prodNCContainer.load();
        SolrClient prodDescIndex_new = new EmbeddedSolrServer(prodNCContainer.getCore("proddesc"));


        ProdDescExporter_NewSchema_Lucene exporter = new ProdDescExporter_NewSchema_Lucene(0,
                jobStart, jobs,
                prodNameDescIndex,
                5000,prodDescIndex_new
                );


        exporter.export();
        prodDescIndex_new.close();
        prodNameDescIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }
}
