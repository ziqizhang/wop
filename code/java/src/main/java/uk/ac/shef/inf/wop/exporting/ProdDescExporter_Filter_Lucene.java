package uk.ac.shef.inf.wop.exporting;

import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.FacetField;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.core.CoreContainer;
import org.apache.solr.core.SolrCore;
import org.apache.solr.search.SolrIndexSearcher;

import java.io.IOException;
import java.net.URL;
import java.util.*;

/**
 * processes the index created by ProdDescExporter_NewSchema_Lucene to export data to a new
 * index of the same schema, but only contain hosts meeting certain criteria
 *
 */
public class ProdDescExporter_Filter_Lucene {
    private List<String> invalidDomains = Arrays.asList(".ru", ".rs", ".gr", ".pl", ".md", ".fr",
            ".ro", ".dk", ".ua", ".at", ".bg", ".tw", ".by", ".hk", ".it", ".jp", ".in", ".no", ".lt", ".hu",
            ".ch", ".ir", ".kz", ".mx", ".su", ".br",
            ".cz", ".ee", ".sk", ".si", ".be", ".de", ".nl", ".es");

    private static final Logger LOG = Logger.getLogger(ProdDescExporter_Filter_Lucene.class.getName());

    private int id;
    private int start;
    private int end;
    private IndexReader luceneIndexReader;
    private SolrClient prodNameDescIndex;
    private int resultBatchSize;

    public ProdDescExporter_Filter_Lucene(int id, int start, int end,
                                             IndexReader prodNameDescIndex_old,
                                             int resultBatchSize, SolrClient prodNameDescIndex_new) {
        this.id = id;
        this.start = start;
        this.end = end;
        this.luceneIndexReader = prodNameDescIndex_old;
        this.resultBatchSize = resultBatchSize;
        this.prodNameDescIndex = prodNameDescIndex_new;
    }

    public void export(List<String> validHosts) {
        long total = 0;

        LOG.info(String.format("\tthread %d: Started, begin=%d end=%d...",
                id, start, end));

        try {

            long countAdded = 0;
            //update results
            LOG.info(String.format("\t\ttotal=%d", luceneIndexReader.maxDoc()));
            for (int i = start; i < luceneIndexReader.maxDoc() && i < end; i++) {

                Document doc = luceneIndexReader.document(i);
                try {
                    boolean added = exportRecord(doc, prodNameDescIndex, validHosts);
                    if (added)
                        countAdded++;
                    if (i % resultBatchSize == 0) {
                        LOG.info(String.format("\t\tthread %d: total results of %d, currently processing %d /started at %d to %d...",
                                id, total, i, start, end));
                        prodNameDescIndex.commit();
                    }
                }catch (Exception e){
                    System.err.println("\t\t error encountered, skipped");
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
                                 SolrClient newIndex, List<String> validHosts) throws IOException, SolrServerException {

        String id = d.get("id");
        String nameData = d.get("name");
        String descData = d.get("text");
        String host="";

        String url=id.split("\\|")[1].trim();
        try{
            URL u = new URL(url);
            host=u.getHost();
            if (!validHosts.contains(host)||!isValidHost(host))
                return false;
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
        newD.addField("id", id);
        newD.addField("host", host);
        newD.addField("name", nameData);
        newD.addField("text", descData);
        newIndex.add(newD);
        return true;
    }

    private boolean isValidHost(String host) {
        for (String d : invalidDomains) {
            if (host.endsWith(d))
                return false;
        }

        return true;
    }

    private String cleanData(String value) {
        value = StringEscapeUtils.unescapeJava(value);
        value = value.replaceAll("\\s+", " ");
        value = StringUtils.stripAccents(value);
        return value.trim();
    }

    private static List<String> findValidHosts(SolrClient index, int minProducts, int topN) throws IOException, SolrServerException {
        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setFacet(true);
        query.setFacetLimit(-1);
        query.setFacetMinCount(minProducts);
        query.addFacetField("host");

        QueryResponse qr = index.query(query);
        FacetField ff = qr.getFacetField("host");
        List<String> hosts = new ArrayList<>();
        Map<String, Long> freq=new HashMap<>();

        for (FacetField.Count c : ff.getValues()) {
            freq.put(c.getName(), c.getCount());
            hosts.add(c.getName());
        }

        hosts.sort((s, t1) -> freq.get(t1).compareTo(freq.get(s)));

        List<String> selected=new ArrayList<>();
        for (int i=0;i<topN && i<hosts.size();i++)
            selected.add(hosts.get(i));

        return selected;
    }


    public static void main(String[] args) throws IOException, SolrServerException {

        //74488335
        int jobStart = Integer.valueOf(args[2]);
        int jobs = Integer.valueOf(args[3]);

        int minFacetCount=100;
        int topN=Integer.MAX_VALUE;

        CoreContainer prodNDContainer = new CoreContainer(args[0]);
        prodNDContainer.load();
        SolrCore core=prodNDContainer.getCore("proddesc");
        SolrClient prodDescIndex = new EmbeddedSolrServer(core);
        List<String> validHosts = ProdDescExporter_Filter_Lucene.findValidHosts(prodDescIndex,
                minFacetCount, topN);
        //prodDescIndex.close();

       // prodNDContainer = new CoreContainer(args[0]);
        //prodNDContainer.load();

        SolrIndexSearcher solrIndexSearcher = core.getSearcher().get();
        IndexReader prodNameDescIndex = solrIndexSearcher.getIndexReader();

        CoreContainer prodNCContainer = new CoreContainer(args[1]);
        prodNCContainer.load();
        SolrClient prodDescIndex_filtered = new EmbeddedSolrServer(prodNCContainer.getCore("proddesc"));


        ProdDescExporter_Filter_Lucene exporter = new ProdDescExporter_Filter_Lucene(0,
                jobStart, jobs,
                prodNameDescIndex,
                5000,prodDescIndex_filtered
        );


        exporter.export(validHosts);
        prodDescIndex_filtered.close();
        prodNameDescIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }

}
