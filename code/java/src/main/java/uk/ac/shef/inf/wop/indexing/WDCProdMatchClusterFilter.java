package uk.ac.shef.inf.wop.indexing;

import com.opencsv.CSVWriter;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.text.similarity.JaccardSimilarity;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.FacetField;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.wop.goldstandard.GS1AnnotationFileCreator;
import uk.ac.shef.inf.wop.goldstandard.GS1AnnotationUtil;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

/**
 * - query the index created by WDCProdMatchDatabaseIndexer, find cluster with size > x
 * - (how to select these clusters?)
 *      - for each cluster
 *          for each product
 *              search in index the ONE closest matching product (by name, using sim threshold)
 *              take its category and put it in the shared set
 *          output the shared set of categories for manual analysis (product names too?)
 *          also output the chosen products as log
 *
 */
public class WDCProdMatchClusterFilter {

    private static final Logger LOG = Logger.getLogger(WDCProdMatchClusterFilter.class.getName());
    private JaccardSimilarity stringSim = new JaccardSimilarity();
    private double minSim=0.9;

    public static void main(String[] args) throws Exception {
        String clusterMetadataFile = args[0];

        WDCProdMatchDatasetIndexer indexer = new WDCProdMatchDatasetIndexer();
        indexer.readClusterMetadataLines(clusterMetadataFile);
        List<long[]> clusters=indexer.exportClusterWithMultipleProducts();

        CoreContainer solrContainer1 = new CoreContainer(args[1]);
        solrContainer1.load();
        SolrClient prodCatDescIndex = new EmbeddedSolrServer(solrContainer1.getCore("prodcatdesc"));

        CoreContainer solrContainer2 = new CoreContainer(args[2]);
        solrContainer2.load();
        SolrClient prodMatchIndex = new EmbeddedSolrServer(solrContainer2.getCore("prodmatch"));

        WDCProdMatchClusterFilter processor = new WDCProdMatchClusterFilter();
        processor.process(prodMatchIndex, prodCatDescIndex, clusters, args[3]);
        prodCatDescIndex.close();
        prodMatchIndex.close();
        System.exit(0);

    }

    public void process(SolrClient prodMatchIndex, SolrClient prodCatDescIndex,
                        List<long[]> cluster, String outFolder) throws IOException {
        CSVWriter csvWriter = GS1AnnotationUtil.getCSVWriter(outFolder + "/category_clusters.csv");
        csvWriter.writeNext(new String[]{"Cluster ID", //0
                "Category"
        });

        LOG.info(String.format("Total clusters to process=%d", cluster.size()));
        int count=1;
        for (long[] c : cluster){
            String clusterID=String.valueOf(c[0]);
            long size=c[1];

            LOG.info(String.format("\t processing %d, cluster id=%s size=%d", count, clusterID, size));

            Set<String> products = getProductNames(clusterID,prodMatchIndex);

            //now for each product, query prodCatDescIndex, get the matching product if it has a category
            try {
                Set<String> clusterOfCategories = getCategories(clusterID, outFolder, prodCatDescIndex,
                        products);
                List<String> sorted = new ArrayList<>(clusterOfCategories);
                Collections.sort(sorted);
                for (String cat : sorted){
                    csvWriter.writeNext(new String[]{clusterID, cat});
                }
            } catch (IOException e) {
                LOG.warn(String.format("\t\t error while processing cluster %s due to %s",clusterID,
                        ExceptionUtils.getFullStackTrace(e)));
            }
        }
        csvWriter.close();
    }

    /**
     *
     * @param clusterID
     * @param outFolder
     * @return
     */
    private Set<String> getCategories(String clusterID, String outFolder, SolrClient prodCatDescIndex,
                                      Set<String> products) throws IOException {
        Set<String> categories= new HashSet<>();
        List<String[]> records= new ArrayList<>();
        for(String p: products){
            LOG.info(String.format("\t\t checking product=%s", p));
            SolrQuery q = new SolrQuery();
            q.setQuery("name:("+p+") AND category:*");
            q.setStart(0);
            q.setRows(50);
            QueryResponse res;
            long total = 0;
            try {
                res = prodCatDescIndex.query(q);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal products in cluster=%d...",
                        total));

                for (SolrDocument d : res.getResults()) {
                    String name=d.getFieldValue("name").toString();
                    double sim=stringSim.apply(p, name);
                    if (sim>minSim){
                        String cat = d.getFieldValue("category").toString();
                        categories.add(cat);
                        records.add(new String[]{cat,name});
                    }
                }

            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        //output
        records.sort(Comparator.comparing(t -> t[0]));
        CSVWriter csvWriter = GS1AnnotationUtil.getCSVWriter(outFolder + "/cluster_"+clusterID+".csv");
        csvWriter.writeNext(new String[]{"Category", //0
                "Product Name"
        });
        for (String[] rec: records){
            csvWriter.writeNext(rec);
        }
        csvWriter.close();

        return categories;
    }

    /**
     * Given the cluster id, get all product names belonging to this cluster from the prod match index
     * @param clusterID
     * @return
     */
    private Set<String> getProductNames(String clusterID, SolrClient prodMatchIndex) {
        Set<String> names = new HashSet<>();
        SolrQuery q = new SolrQuery();
        q.setQuery("cluster_id:"+clusterID);
        q.setFacet(true);
        q.addFacetField("title_str");

        q.setStart(0);
        q.setRows(10000);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        long count=0;

        while (!stop) {
            try {
                res = prodMatchIndex.query(q);
                if (res != null)
                    total = res.getResults().getNumFound();
                else
                    break;

                //update results
                LOG.info(String.format("\t\ttotal products in cluster=%d...",
                        total));

                FacetField ff = res.getFacetField("title_str");
                for(FacetField.Count c : ff.getValues()){
                    names.add(c.getName());
                }

            } catch (Exception e) {
                LOG.warn(String.format("\t\t unable to query due to error: %s",
                        ExceptionUtils.getFullStackTrace(e)));

            }

            int curr = q.getStart() + q.getRows();
            if (curr < total)
                q.setStart(curr);
            else
                stop = true;
        }

        return names;
    }
}
