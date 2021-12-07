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
import org.apache.solr.client.solrj.util.ClientUtils;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.core.CoreContainer;
import org.apache.solr.core.SolrCore;
import org.apache.solr.search.SolrIndexSearcher;
import uk.ac.shef.inf.wop.indexing.ProdCatDescIndexCreator;

import java.io.IOException;
import java.net.URL;
import java.util.*;

/**
 * needs the previously cat index and desc index separately, merge them into a single one with the
 * same schema as the new format
 */

public class ProdDescCatIndecesMerger {

    private static final Logger LOG = Logger.getLogger(ProdDescCatIndecesMerger.class.getName());

    private int id;
    private IndexReader luceneIndexReader;
    private SolrClient prodNameCatIndex;
    private SolrClient newIndex;
    private int resultBatchSize;

    public ProdDescCatIndecesMerger(int id,
                                    IndexReader prodNameDescIndex_old,
                                    int resultBatchSize, SolrClient prodNameCatIndex,
                                    SolrClient newIndex) {
        this.id = id;

        this.luceneIndexReader = prodNameDescIndex_old;
        this.resultBatchSize = resultBatchSize;
        this.prodNameCatIndex = prodNameCatIndex;
        this.newIndex = newIndex;
    }

    public void merge() {
        long total = 0;

        try {

            long countAdded = 0;
            //update results
            LOG.info(String.format("\t\ttotal=%d", luceneIndexReader.maxDoc()));
            for (int i = 0; i < luceneIndexReader.maxDoc(); i++) {

                Document doc = luceneIndexReader.document(i);
                try {
                    boolean added = mergeRecord(doc, prodNameCatIndex, newIndex);
                    if (added)
                        countAdded++;
                    if (i % resultBatchSize == 0) {
                        LOG.info(String.format("\t\tthread %d: total results of %d, currently processing %d ...",
                                id, total, i));
                        newIndex.commit();
                    }
                } catch (Exception e) {
                    System.err.println("\t\t error encountered, skipped");
                }
                // do something with docId here...

            }

            newIndex.commit();
        } catch (Exception ioe) {
            LOG.warn(String.format("\t\t thread %d unable to create output files, io exception: %s",
                    id, ExceptionUtils.getFullStackTrace(ioe)));
        }
    }

    private boolean mergeRecord(Document d,
                                SolrClient prodNameCatIndex,
                                SolrClient newIndex) throws IOException, SolrServerException {

        String id = d.get("id");
        String nameData = d.get("name");
        String descData = d.get("text");
        String host = "";

        String url = id.split("\\|")[1].trim();
        try {
            URL u = new URL(url);
            host = u.getHost();
            if (!ProdCatDescIndexCreator.checkHost(host))
                return false;
        } catch (Exception e) {
            System.err.println("\t\t encountered invalid url in id:" + id);
            return false;
        }

        SolrDocument recordInNameCat = findRecord(prodNameCatIndex, id);

        SolrInputDocument newD = new SolrInputDocument();
        newD.addField("id", id);
        newD.addField("host", host);
        newD.addField("url", url);
        newD.addField("name", nameData);
        newD.addField("desc", descData);
        newD.addField("source_entity_index", "-1");
        if (recordInNameCat != null) {
            String cat = recordInNameCat.getFieldValue("category_str").toString();
            newD.addField("category_str", cat);
            newD.addField("category", cat);
        }

        newIndex.add(newD);
        return true;
    }

    private SolrDocument findRecord(SolrClient prodNameCatIndex, String id) {
        SolrQuery query = new SolrQuery();
        query.setQuery("id:" + ClientUtils.escapeQueryChars(id));
        query.setStart(0);
        query.setRows(10);
        try {
            QueryResponse res = prodNameCatIndex.query(query);
            if (res != null && res.getResults().getNumFound() > 0) {
                return res.getResults().get(0);
            }
        } catch (Exception e) {
            System.err.println(String.format("failed to find record %s in the name-cat index", id));
            e.printStackTrace();
        }
        return null;
    }


    public static void main(String[] args) throws IOException, SolrServerException {

        CoreContainer oldprodDescContainer = new CoreContainer(args[0]);
        oldprodDescContainer.load();
        SolrCore core1 = oldprodDescContainer.getCore("proddesc");

        CoreContainer prodNCContainer = new CoreContainer(args[1]);
        prodNCContainer.load();
        SolrCore core2 = prodNCContainer.getCore("prodcat");
        SolrClient prodCatIndex = new EmbeddedSolrServer(core2);

        CoreContainer newIndexContainer = new CoreContainer(args[2]);
        newIndexContainer.load();
        SolrCore core3 = newIndexContainer.getCore("prodcatdesc");
        SolrClient newIndex = new EmbeddedSolrServer(core3);
        //prodDescIndex.close();

        // prodNDContainer = new CoreContainer(args[0]);
        //prodNDContainer.load();

        SolrIndexSearcher solrIndexSearcher = core1.getSearcher().get();
        IndexReader oldProdDescIndex = solrIndexSearcher.getIndexReader();

        ProdDescCatIndecesMerger exporter = new ProdDescCatIndecesMerger(0,
                oldProdDescIndex, 100000, prodCatIndex, newIndex
        );

        exporter.merge();
        newIndex.close();
        oldProdDescIndex.close();
        prodCatIndex.close();
        System.exit(0);
        LOG.info("COMPLETE!");

    }

}
