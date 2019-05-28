package uk.ac.shef.inf.wop.exporting;

import com.google.common.base.Optional;
import com.optimaize.langdetect.LanguageDetector;
import com.optimaize.langdetect.LanguageDetectorBuilder;
import com.optimaize.langdetect.i18n.LdLocale;
import com.optimaize.langdetect.ngram.NgramExtractors;
import com.optimaize.langdetect.profiles.LanguageProfile;
import com.optimaize.langdetect.profiles.LanguageProfileReader;
import com.optimaize.langdetect.text.CommonTextObjectFactories;
import com.optimaize.langdetect.text.TextObject;
import com.optimaize.langdetect.text.TextObjectFactory;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.core.CoreContainer;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * this classes runs on top of an index created by NTripleIndexerApp, and exports product name-category paris into a solr index
 * subject to some criteria
 */
public class ProdNameCategoryExporter {

    private static final Logger LOG = Logger.getLogger(ProdNameCategoryExporter.class.getName());

    private static List<String> stopwords= Arrays.asList("product","home");

    private LanguageDetector languageDetector;
    private TextObjectFactory textObjectFactory;
    public ProdNameCategoryExporter() throws IOException {
        //load all languages:
        List<LanguageProfile> languageProfiles = new LanguageProfileReader().readAllBuiltIn();

//build language detector:
        languageDetector = LanguageDetectorBuilder.create(NgramExtractors.standard())
                .withProfiles(languageProfiles)
                .build();

//create a text object factory
        textObjectFactory = CommonTextObjectFactories.forDetectingOnLargeText();

    }

    /*
    (rdfs_type:"http://schema.org/Product") AND (sg-product_name:*) AND (sg-product_category:*)

( rdfs_type:"http://schema.org/Offer") AND (sg-offer_name:*) AND (sg-offer_category:*)
     */
    private SolrQuery createQuery(int resultBatchSize, int start){
        SolrQuery query = new SolrQuery();
        query.setQuery("(rdfs_type:\"http://schema.org/Product\" OR rdfs_type:\"http://schema.org/Offer\") " +
                "AND (sg-product_name:* OR sg-offer_name:*) AND (sg-product_category:* OR sg-offer_category:*)");
        query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    private void export(SolrClient prodTripleIndex, int resultBatchSize,
                        SolrClient prodNameCatIndex){
        int start=0;
        SolrQuery q = createQuery(resultBatchSize,start);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        long count=0;

        while (!stop) {
            try {
                res = prodTripleIndex.query(q);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));

                for (SolrDocument d : res.getResults()) {
                    //process and export to the other solr index
                    int added = createRecord(d, prodNameCatIndex);
                    count+=added;
                }

                start = start + resultBatchSize;
                prodNameCatIndex.commit();
                LOG.info(String.format("\t\ttotal indexed = %d.",
                        count));
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

        try{
            prodTripleIndex.close();
            prodNameCatIndex.commit();
            prodNameCatIndex.close();
        }catch (Exception e){
            LOG.warn(String.format("\t\t unable to shut down servers due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    /*
    DiMarzio DP223F PAF 36th Anniversary Humbucker Pickup, F-Spaced, Bridge, Black|Pickups
discarded pair=Dunlop 535Q Cry Baby Multi-Wah Pedal|Guitar Pedals | Effects Pedals
     */
    private int createRecord(SolrDocument d, SolrClient prodcatIndex) throws IOException, SolrServerException {
        String id = d.getFieldValue("id").toString();
        Object h = d.getFieldValue("source_host").toString();
        String url =d.getFieldValue("source_page").toString();
        SolrInputDocument doc = new SolrInputDocument();
        int added=0;
        if (d.getFieldValue("sg-product_name")!=null) {
            String name = d.getFieldValue("sg-product_name").toString().replaceAll("\\s+"," ").trim();
            /*if (d.getFieldValue("sg-product_name").toString().equalsIgnoreCase("Dunlop 535Q Cry Baby Multi-Wah Pedal"))
                System.out.println();*/
            if (d.getFieldValue("sg-product_category")==null)
                return 0;
            String cat = d.getFieldValue("sg-product_category").toString().replaceAll("\\s+"," ").trim();;
            name=cleanData(name);
            cat=cleanData(cat);
            if (name==null ||cat==null) {
                //System.out.println("discarded pair="+d.getFieldValue("sg-product_name").toString()+"|"+d.getFieldValue("sg-product_category").toString());
                return 0;
            }

            if (cat.startsWith(name))
                cat=cat.substring(name.length()).trim();
            if (cat.length()<3)
                return 0;
            doc.setField("id",id);
            doc.setField("name",name);
            doc.setField("category",cat);
            doc.setField("category_str",cat);
            doc.setField("host",h);
            doc.setField("url",url);
            prodcatIndex.add(doc);
            added++;
        }
        if(d.getFieldValue("sg-offer_name")!=null) {
            String name=d.getFieldValue("sg-offer_name").toString().replaceAll("\\s+"," ").trim();;
            /*if (d.getFieldValue("sg-offer_name").toString().equalsIgnoreCase("Dunlop 535Q Cry Baby Multi-Wah Pedal"))
                System.out.println();*/
            if (d.getFieldValue("sg-offer_category")==null)
                return 0;
            String cat = d.getFieldValue("sg-offer_category").toString().replaceAll("\\s+"," ").trim();;
            name=cleanData(name);
            cat=cleanData(cat);
            if (name==null ||cat==null) {
                //System.out.println("discarded pair="+d.getFieldValue("sg-offer_name").toString()+"|"+d.getFieldValue("sg-offer_category").toString());
                return 0;
            }
            if (cat.startsWith(name))
                cat=cat.substring(name.length()).trim();
            if (cat.length()<3)
                return 0;

            doc.setField("id",id+"_sgoffer");
            doc.setField("name",name);
            doc.setField("category",cat);
            doc.setField("category_str",cat);
            doc.setField("host",h);
            prodcatIndex.add(doc);
            added++;
        }
        else {
            return 0;
        }
        return added;
    }

    /**
     * implements rules to clean values in the product name and category fields
     * @param value
     * @return
     */
    private String cleanData(String value) {
        value=value.trim();

        if (stopwords.contains(value.toLowerCase()))
            return null;

        if (value.length()<5)
            return null;
        /*if (value.split("\\s+").length<2)
            return null;*/

        if (value.startsWith(".")||value.startsWith("\\u"))
            return null;
        TextObject textObject = textObjectFactory.forText(value);
        Optional<LdLocale> lang = languageDetector.detect(textObject);
        if (lang.isPresent()&&!lang.get().getLanguage().equalsIgnoreCase("en"))
            return null;
        return value;
    }

    public static void main(String[] args) throws IOException {
        CoreContainer prodIndexContainer = new CoreContainer(args[0]);
        prodIndexContainer.load();
        SolrClient prodTripleIndex = new EmbeddedSolrServer(prodIndexContainer.getCore("entities"));

        CoreContainer prodNCContainer = new CoreContainer(args[1]);
        prodNCContainer.load();
        SolrClient prodNameCatIndex = new EmbeddedSolrServer(prodNCContainer.getCore("prodcat"));

        ProdNameCategoryExporter exporter = new ProdNameCategoryExporter();
        exporter.export(prodTripleIndex, Integer.valueOf(args[2]), prodNameCatIndex);
        LOG.info("COMPLETE!");

        System.exit(0);

    }
}
