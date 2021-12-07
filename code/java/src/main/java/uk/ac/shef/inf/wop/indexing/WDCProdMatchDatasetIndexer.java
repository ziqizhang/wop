package uk.ac.shef.inf.wop.indexing;

import com.google.gson.*;
import com.google.gson.stream.JsonReader;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.math3.stat.descriptive.rank.Max;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.core.CoreContainer;
import org.mapdb.DB;
import org.mapdb.DBMaker;
import org.mapdb.Serializer;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * This produces the JSON file from: http://webdatacommons.org/largescaleproductcorpus/v2/index.html#toc1
 * (Product Data Corpus (English) sample_offers_english 4.0GB offers_corpus_english_v2.json.gz)
 * <p>
 * and index the dataset into a solr index. It will keep all fields defined on the webpage under section
 * '2.2. Schema of the Corpus'.
 * <p>
 * In total, stats should be '16 million offers into 10 million ID-clusters'
 */

public class WDCProdMatchDatasetIndexer {
    private static final Logger LOG = Logger.getLogger(WDCProdMatchDatasetIndexer.class.getName());

    public static void main(String[] args) throws Exception {
        //code for testing cluster info reading
        /*String clusterMetadataFile = args[0];
        WDCProdMatchDatasetIndexer indexer = new WDCProdMatchDatasetIndexer();
        indexer.readClusterMetadataLines(clusterMetadataFile);
        System.exit(0);*/

        //code for indexing products
        CoreContainer prodNCContainer = new CoreContainer(args[0]);
        prodNCContainer.load();
        SolrClient index = new EmbeddedSolrServer(prodNCContainer.getCore("prodmatch"));
        WDCProdMatchDatasetIndexer indexer = new WDCProdMatchDatasetIndexer();
        String clusterMetadataFile = args[1];
        indexer.readClusterMetadataLines(clusterMetadataFile);
        System.out.println("Started indexing products...");
        indexer.indexProducts(args[2], index);
        index.close();
        System.exit(0);
    }

    private DB db = DBMaker.fileDB("tmp.db")
            .fileMmapEnable()
            .allocateStartSize(12 * 1024 * 1024)  // 1GB
            .allocateIncrement(512 * 1024 * 1024)       // 512MB
            .make();

    //int 0: cluster size as # of different products; int 1: cluster size as # of different product ids
    private Map<String, long[]> clusterSize
            = db.hashMap("ClusterSize", Serializer.STRING, Serializer.LONG_ARRAY).createOrOpen();


    /**
     * Process the cluster metadata file (idclusters.json) and save cluster id and its size information
     *
     * @param jsonFile
     */
    public void readClusterMetadata(String jsonFile) {
        try (JsonReader jsonReader = new JsonReader(
                new InputStreamReader(
                        new FileInputStream(jsonFile), StandardCharsets.UTF_8))) {

            Gson gson = new GsonBuilder().create();

            jsonReader.beginObject(); //start of json array
            int numberOfRecords = 0;
            while (jsonReader.hasNext()) { //next json array element
                Cluster c = gson.fromJson(jsonReader, Cluster.class);
                long[] sizeInfo = new long[]{c.cluster_size_in_offers, c.size};
                clusterSize.put(String.valueOf(c.id), sizeInfo);

                numberOfRecords++;
                if (numberOfRecords % 100000 == 0)
                    System.out.println(String.format("processed %d clusters", numberOfRecords));
            }
            jsonReader.endArray();
            System.out.println("Total Records Found : " + numberOfRecords);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    /**
     * Process the cluster metadata file (idclusters.json) and save cluster id and its size information
     *
     * @param jsonFile
     */
    public void readClusterMetadataLines(String jsonFile) throws Exception {
        BufferedReader in = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream(jsonFile), "UTF8"));
        Gson gson = new GsonBuilder().create();
        String str;
        int numberOfRecords = 0;
        while ((str = in.readLine()) != null) {

            Cluster c = gson.fromJson(str, Cluster.class);
            long[] sizeInfo = new long[]{c.cluster_size_in_offers, c.size};
            clusterSize.put(String.valueOf(c.id), sizeInfo);

            numberOfRecords++;
            if (numberOfRecords % 100000 == 0)
                System.out.println(String.format("processed %d clusters", numberOfRecords));
        }
        in.close();
        System.out.println("Total Records Found : " + numberOfRecords);
    }

    public List<long[]> exportClusterWithMultipleProducts() throws Exception {
        List<long[]> out = new ArrayList<>();
        for (Map.Entry<String, long[]> e : clusterSize.entrySet()){
            if (e.getValue()[0]>1)
                out.add(new long[]{Long.valueOf(e.getKey()),e.getValue()[0]});
        }

        Collections.sort(out, (t1, t2) -> Long.compare(t2[1], t1[1]));
        return out;
    }

    /**
     * Process the product metadata file (offers_corpus_english_v2.json) and index each product
     */
    public void indexProducts(String jsonFile, SolrClient solrIndex) throws IOException {
        int batchSize = 100000;
        BufferedReader in = new BufferedReader(
                new InputStreamReader(
                        new FileInputStream(jsonFile), "UTF8"));
        Gson gson = new GsonBuilder().create();
        String str;
        int numberOfRecords = 0, start = 0;
        try {
            while ((str = in.readLine()) != null) {
                //System.out.println(str);
                Product p = gson.fromJson(str, Product.class);
                try {
                    indexProduct(p, solrIndex);
                } catch (Exception e) {
                    LOG.warn(String.format("\t\t unable to add product instance, line %d " +
                                    "Due to error: %s",
                            numberOfRecords,
                            ExceptionUtils.getFullStackTrace(e)));
                }

                numberOfRecords++;
                if (numberOfRecords % batchSize == 0) {
                    System.out.println(String.format("processed %d products", numberOfRecords));
                    try {
                        solrIndex.commit();
                    } catch (SolrServerException e) {
                        LOG.warn(String.format("\t\t unable to commit index product data starting from index %d, with batch size %d. " +
                                        "Due to error: %s",
                                start, batchSize,
                                ExceptionUtils.getFullStackTrace(e)));

                    }
                    start = numberOfRecords;
                }
            }
        } catch (IOException ioe) {
            LOG.warn(String.format("\t\t failed to read json file at line %d, due to error %s",
                    numberOfRecords, ExceptionUtils.getFullStackTrace(ioe)));
        }
        in.close();
        System.out.println("Total Records Found : " + numberOfRecords);
    }

    private void indexProduct(Product p, SolrClient solrIndex) throws IOException, SolrServerException {
        SolrInputDocument doc = new SolrInputDocument();

        long[] cluster_size_info = clusterSize.get(p.cluster_id);
        doc.addField("id", p.id);
        doc.addField("cluster_id", p.cluster_id);
        doc.addField("category", p.category);
        doc.addField("description", p.description);
        doc.addField("brand", p.brand);
        doc.addField("price", p.price);
        doc.addField("title",p.title);
        doc.addField("title_str",p.title);
        doc.addField("specTableContent", p.specTableContent);
        if (cluster_size_info==null){
            LOG.warn(String.format("\t\t\t product id=%s has cluster_id=%s, but this cluster_id is not found in the cluster metadata",
                    p.id, p.cluster_id));
        }
        else{
            doc.addField("cluster_size",cluster_size_info[1]);
            doc.addField("cluster_size_in_offers",cluster_size_info[0]);
        }

        //identifiers
        String identifiers=p.identifiers.toString().replaceAll("[\\[\\]{}\"]","");
        List<String> ids = Arrays.asList(identifiers.split(","));
        doc.addField("identifiers",ids);


        //key value pairs
        if (p.keyValuePairs instanceof JsonNull)
            doc.addField("keyValuePairs",null);
        else {
            String keyvalue = p.keyValuePairs.toString().replaceAll("[\\[\\]{}\"]", "");
            List<String> kvs = Arrays.asList(keyvalue.split(","));
            doc.addField("keyValuePairs", kvs);
        }
        solrIndex.add(doc);

    }

    private class Cluster {
        protected long id;
        protected long size;
        protected long cluster_size_in_offers;
        protected String id_values;
        protected double categoryDensity;
        protected String category;

        public Cluster() {
        }

        public Cluster(JsonObject json) {
            this.id = json.getAsJsonObject("id").getAsLong();
            this.size = json.getAsJsonObject("size").getAsLong();
            this.cluster_size_in_offers = json.getAsJsonObject("cluster_size_in_offers").getAsLong();
            this.id_values = json.getAsJsonObject("id_values").getAsString();
            this.categoryDensity = json.getAsJsonObject("categoryDensity").getAsDouble();
            this.category = json.getAsJsonObject("category").getAsString();
        }

    }


    private class Product {
        protected long id;
        protected String cluster_id;
        protected JsonArray identifiers;
        protected String category;
        protected String title;
        protected String description;
        protected String brand;
        protected String price;
        protected String specTableContent;
        protected JsonElement keyValuePairs;

        public Product() {
        }

        public Product(JsonObject json) {
            this.id = json.get("id").getAsLong();
            this.cluster_id = json.get("cluster_id").getAsString();
            this.identifiers = json.getAsJsonArray("identifiers");
            this.category = json.get("category").getAsString();
            this.title = json.get("title").getAsString();
            this.description = json.get("description").getAsString();
            this.brand = json.get("brand").getAsString();
            this.price = json.get("price").getAsString();
            this.keyValuePairs = json.get("keyValuePairs");
            this.specTableContent = json.get("specTableContent").getAsString();
        }

    }
}
