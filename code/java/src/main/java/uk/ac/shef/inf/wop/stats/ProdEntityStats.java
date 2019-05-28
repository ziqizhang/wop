package uk.ac.shef.inf.wop.stats;

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
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;

import java.io.IOException;
import java.util.*;

/**
 * calculate some stats for the entity_# indeces
 * <p>
 * - gtin12, 13, 14, 8, mpn, nsn, sku, productID
 * - how many duplicates,
 * - for duplicates, how many hosts
 * - ALL THESE must have categories
 */
public class ProdEntityStats {

    private static final Logger LOG = Logger.getLogger(ProdEntityStats.class.getName());

    private static List<String> stopwords = Arrays.asList("product", "home");
    private static List<String> prodIDFields=
            Arrays.asList("https://schema.org/Product/gtin8_t",
                    "https://schema.org/Product/gtin12_t",
                    "https://schema.org/Product/gtin13_t",
                    "https://schema.org/Product/gtin14_t",
                    "https://schema.org/Product/mpn",
                    "https://schema.org/Product/nsn",
                    "https://schema.org/Product/sku",
                    "https://schema.org/Product/productID",
                    "http://schema.org/Product/gtin8_t",
                    "http://schema.org/Product/gtin12_t",
                    "http://schema.org/Product/gtin13_t",
                    "http://schema.org/Product/gtin14_t",
                    "http://schema.org/Product/mpn",
                    "http://schema.org/Product/nsn",
                    "http://schema.org/Product/sku",
                    "http://schema.org/Product/productID",
                    "https://schema.org/Offer/gtin8_t",
                    "https://schema.org/Offer/gtin12_t",
                    "https://schema.org/Offer/gtin13_t",
                    "https://schema.org/Offer/gtin14_t",
                    "https://schema.org/Offer/mpn",
                    "https://schema.org/Offer/nsn",
                    "https://schema.org/Offer/sku",
                    "https://schema.org/Offer/productID",
                    "http://schema.org/Offer/gtin8_t",
                    "http://schema.org/Offer/gtin12_t",
                    "http://schema.org/Offer/gtin13_t",
                    "http://schema.org/Offer/gtin14_t",
                    "http://schema.org/Offer/mpn",
                    "http://schema.org/Offer/nsn",
                    "http://schema.org/Offer/sku",
                    "http://schema.org/Offer/productID");

    private LanguageDetector languageDetector;
    private TextObjectFactory textObjectFactory;
    private Map<String, Set<String>> sameProdIDs=new HashMap<>();

    private long removeLang=0;
    private long removeStopwords=0;
    private long removeShort =0;
    private long removeUnicode=0;

    public ProdEntityStats() throws IOException {
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
    private SolrQuery createQuery(int resultBatchSize, int start) {
        SolrQuery query = new SolrQuery();
        query.setQuery("(rdfs_type:\"http://schema.org/Product\" OR rdfs_type:\"http://schema.org/Offer\") " +
                "AND (sg-product_name:* OR sg-offer_name:*) AND (sg-product_category:* OR sg-offer_category:*)");
        query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    private void calcStats(SolrClient prodTripleIndex, int resultBatchSize
                           ) {
        int start = 0;
        SolrQuery q = createQuery(resultBatchSize, start);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        long count = 0;

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
                    int added = updateStats(d);
                    count += added;
                }

                start = start + resultBatchSize;
                LOG.info(String.format("\t\ttotal checked = %d.",
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

        try {
            prodTripleIndex.close();
        } catch (Exception e) {
            LOG.warn(String.format("\t\t unable to shut down servers due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    /*
    DiMarzio DP223F PAF 36th Anniversary Humbucker Pickup, F-Spaced, Bridge, Black|Pickups
discarded pair=Dunlop 535Q Cry Baby Multi-Wah Pedal|Guitar Pedals | Effects Pedals
     */
    private int updateStats(SolrDocument d) {
        int added = 0;
        String host = d.get("source_host").toString();
        if (d.getFieldValue("sg-product_name") != null) {
            String name = d.getFieldValue("sg-product_name").toString().replaceAll("\\s+", " ").trim();
            /*if (d.getFieldValue("sg-product_name").toString().equalsIgnoreCase("Dunlop 535Q Cry Baby Multi-Wah Pedal"))
                System.out.println();*/
            if (d.getFieldValue("sg-product_category") == null)
                return 0;
            String cat = d.getFieldValue("sg-product_category").toString().replaceAll("\\s+", " ").trim();
            ;
            name ="";// cleanData(name);
            cat = cleanData(cat);
            if (name == null || cat == null) {
                //System.out.println("discarded pair="+d.getFieldValue("sg-product_name").toString()+"|"+d.getFieldValue("sg-product_category").toString());
                return 0;
            }

            if (cat.startsWith(name))
                cat = cat.substring(name.length()).trim();
            if (cat.length() < 3)
                return 0;
            /*
             * - gtin12, 13, 14, 8, mpn, nsn, sku, productID
             * - how many duplicates,
             * - for duplicates, how many hosts
             */
            added++;
        }
        if (d.getFieldValue("sg-offer_name") != null) {
            String name = d.getFieldValue("sg-offer_name").toString().replaceAll("\\s+", " ").trim();
            ;
            /*if (d.getFieldValue("sg-offer_name").toString().equalsIgnoreCase("Dunlop 535Q Cry Baby Multi-Wah Pedal"))
                System.out.println();*/
            if (d.getFieldValue("sg-offer_category") == null)
                return 0;
            String cat = d.getFieldValue("sg-offer_category").toString().replaceAll("\\s+", " ").trim();
            ;
            name ="";// cleanData(name);
            cat = cleanData(cat);
            if (name == null || cat == null) {
                //System.out.println("discarded pair="+d.getFieldValue("sg-offer_name").toString()+"|"+d.getFieldValue("sg-offer_category").toString());
                return 0;
            }
            if (cat.startsWith(name))
                cat = cat.substring(name.length()).trim();
            if (cat.length() < 3)
                return 0;
            added++;
        } else {
            return 0;
        }

        //there is a valid category, now check if there is product id
        if (added!=0){
            Collection<String> idFields=
                    CollectionUtils.intersection(d.getFieldNames(), prodIDFields);
            if (idFields!=null && idFields.size()>0){
                for (String field: idFields){
                    String idStr=d.getFieldValue(field).toString();
                    if (sameProdIDs.containsKey(idStr)){
                        Set<String> hosts =sameProdIDs.get(idStr);
                        if (hosts==null)
                            hosts=new HashSet<>();
                        hosts.add(host);
                        sameProdIDs.put(idStr, hosts);
                    }else{
                        Set<String> hosts=new HashSet<>();
                        hosts.add(host);
                        sameProdIDs.put(idStr, hosts);
                    }
                }
            }
        }
        System.out.println(String.format("\t\t\tremove due to lang=%d, stopwords=%d, " +
                "unicode=%d, short=%d",removeLang,removeStopwords,removeUnicode, removeShort));
        return added;
    }

    /**
     * implements rules to clean values in the product name and category fields
     *
     * @param value
     * @return
     */
    private String cleanData(String value) {
        value = value.trim();

        if (stopwords.contains(value.toLowerCase())) {
            removeStopwords++;
            return null;
        }

        if (value.length() < 5) {
            removeShort++;
            //System.out.println(value);
            return null;
        }
        /*if (value.split("\\s+").length<2)
            return null;*/

        if (value.startsWith(".") || value.startsWith("\\u")) {
            removeUnicode++;
            return null;
        }
        TextObject textObject = textObjectFactory.forText(value);
        Optional<LdLocale> lang = languageDetector.detect(textObject);
        if (lang.isPresent() && !lang.get().getLanguage().equalsIgnoreCase("en")) {
            removeLang++;
            return null;
        }
        return value;
    }

    private void outputStats() {
        System.out.println(sameProdIDs.size());
        int count=0;
        for (Collection<String> c:sameProdIDs.values()){
            if (c.size()>1)
                count++;
        }
        System.out.println(count);
    }

    public static void main(String[] args) throws IOException {
        CoreContainer prodIndexContainer = new CoreContainer(args[0]);
        prodIndexContainer.load();
        SolrClient prodTripleIndex = new EmbeddedSolrServer(prodIndexContainer.getCore("entities"));

        ProdEntityStats statCalc = new ProdEntityStats();
        statCalc.calcStats(prodTripleIndex, Integer.valueOf(args[1]));
        LOG.info("COMPLETE!");
        statCalc.outputStats();

        System.exit(0);

    }

}
