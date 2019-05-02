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
 * this class runs on top of an index created by NTripleIndexerApp, and exports product name-description paris into a solr index
 * subject to some criteria
 */
public class ProdDescExporter {

    private static final Logger LOG = Logger.getLogger(ProdDescExporter.class.getName());

    private static List<String> stopwords= Arrays.asList("product","www");

    private LanguageDetector languageDetector;
    private TextObjectFactory textObjectFactory;
    public ProdDescExporter() throws IOException {
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
                "AND ((sg-product_name:* OR sg-offer_name:*) OR sg-product_description:*)");
        query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    private void export(SolrClient prodTripleIndex, int resultBatchSize,
                        SolrClient prodNameDescIndex){
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
                    int added = createRecord(d, prodNameDescIndex);
                    count+=added;
                }

                start = start + resultBatchSize;
                prodNameDescIndex.commit();
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
            prodNameDescIndex.close();
            prodNameDescIndex.commit();
            prodNameDescIndex.close();
        }catch (Exception e){
            LOG.warn(String.format("\t\t unable to shut down servers due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    private int createRecord(SolrDocument d, SolrClient prodcatIndex) throws IOException, SolrServerException {
        String id = d.getFieldValue("id").toString();
        SolrInputDocument doc = new SolrInputDocument();

        String name="";
        if (d.getFieldValue("sg-product_name")!=null) {
            name = d.getFieldValue("sg-product_name").toString().trim();
        }
        else if(d.getFieldValue("sg-offer_name")!=null) {
            name=d.getFieldValue("sg-offer_name").toString().trim();
        }
        if (stopwords.contains(name.toLowerCase()))
            return 0;
        if (name.length()<10||name.length()>20)
            return 0;
        if (name.split("\\s+").length<5)
            return 0;
        if (name.startsWith(".")||name.startsWith("\\u"))
            return 0;

        doc.setField("id",id);
        doc.setField("name",name);

        if (d.getFieldValue("sg-product_description")!=null){
            String desc=d.getFieldValue("sg-product_description").toString();
            desc=desc.replaceAll("\\s+"," ").trim();
            desc=cleanData(desc);
            if (desc==null)
                return 0;
            doc.setField("text", desc);
        }

        prodcatIndex.add(doc);

        return 1;
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

        if (value.length()<10)
            return null;
        if (value.split("\\s+").length<5)
            return null;

        if (value.startsWith(".")||value.startsWith("\\u"))
            return null;
        TextObject textObject = textObjectFactory.forText(value);
        Optional<LdLocale> lang = languageDetector.detect(textObject);
        if (!lang.isPresent())
            return null;
        if (!lang.get().getLanguage().equalsIgnoreCase("en"))
            return null;
        return value;
    }

    public static void main(String[] args) throws IOException {
        CoreContainer prodIndexContainer = new CoreContainer(args[0]);
        prodIndexContainer.load();
        SolrClient prodTripleIndex = new EmbeddedSolrServer(prodIndexContainer.getCore("entities"));

        CoreContainer prodNDContainer = new CoreContainer(args[1]);
        prodNDContainer.load();
        SolrClient prodNameDescIndex = new EmbeddedSolrServer(prodNDContainer.getCore("proddesc"));

        ProdDescExporter exporter = new ProdDescExporter();
        exporter.export(prodTripleIndex, Integer.valueOf(args[2]), prodNameDescIndex);
        System.exit(0);
        LOG.info("COMPLETE!");

    }
}
