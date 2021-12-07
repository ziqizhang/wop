package uk.ac.shef.inf.wop.indexing;

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
import org.apache.commons.lang.StringEscapeUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.wop.exporting.ProdNameCategoryExporter;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * this class reads the index built by NTripleIndexerApp and builds an index containing the prod category and desc only
 * <p>
 * - id
 * - source index
 * - desc
 * - category
 * - name
 * - host
 * - url
 */
public class ProdCatDescIndexCreator {

    private static final Logger LOG = Logger.getLogger(ProdCatDescIndexCreator.class.getName());

    private static List<String> stopwords = Arrays.asList("product", "home", "null");

    public static List<String> validDomains = Arrays.asList(".uk", ".com", ".net", ".org", ".au", ".ag",
            ".bs", ".bb", ".ca", ".do", ".gd", ".gy", ".ie", ".jm", ".nz", ".kn", ".lc", ".vc", ".tt", ".us");

    private LanguageDetector languageDetector;
    private TextObjectFactory textObjectFactory;

    public ProdCatDescIndexCreator() throws IOException {
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
        /*query.setQuery("(rdfs_type:\"http://schema.org/Product\" OR rdfs_type:\"http://schema.org/Offer\") " +
                "AND (sg-product_name:* OR sg-offer_name:*) AND " +
                "((sg-product_category:* OR sg-offer_category:*) OR sg-product_description:*)");*/
        query.setQuery("(rdfs_type:\"http://schema.org/Product\" OR rdfs_type:\"http://schema.org/Offer\") " +
                "AND (sg-product_name:* OR sg-offer_name:*)");
        query.setStart(start);
        query.setRows(resultBatchSize);

        return query;
    }

    private void export(SolrClient prodTripleIndex, String prodTripleIndexID, int resultBatchSize,
                        SolrClient prodCatDescIndex) {
        int start = 0;
        SolrQuery q = createQuery(resultBatchSize, start);
        QueryResponse res;
        boolean stop = false;
        long total = 0;

        long count = 0;

        LOG.info(String.format("Currently processing index=%s",
                prodTripleIndexID));
        while (!stop) {
            try {
                res = prodTripleIndex.query(q);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));

                for (SolrDocument d : res.getResults()) {
                    //process and export to the other solr index
                    int added = createRecord(prodTripleIndexID, d, prodCatDescIndex, count);
                    count += added;
                }

                start = start + resultBatchSize;
                prodCatDescIndex.commit();

                LOG.info(String.format("\ttotal in this batch = %d, index size now=%d",
                        count, countIndexSize(prodCatDescIndex)));
            } catch (Exception e) {
                LOG.warn(String.format("\t\t unable to successfully index product triples starting from index %d, with batch size %d. Due to error: %s",
                        start, resultBatchSize,
                        ExceptionUtils.getFullStackTrace(e)));

            }

            int curr = q.getStart() + q.getRows();
            if (curr < total)
                q.setStart(curr);
            else
                stop = true;
        }

        try {
            prodCatDescIndex.commit();
            LOG.info(String.format("Recorded=%d, index size=%d",
                    count, countIndexSize(prodCatDescIndex)));
        } catch (Exception e) {
            LOG.warn(String.format("\t\t unable to shut down servers due to error: %s",
                    ExceptionUtils.getFullStackTrace(e)));
        }
    }

    private long countIndexSize(SolrClient solr) throws IOException, SolrServerException {
        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setStart(0);
        query.setRows(10);
        QueryResponse res = solr.query(query);
        return res.getResults().getNumFound();
    }

    /*
    DiMarzio DP223F PAF 36th Anniversary Humbucker Pickup, F-Spaced, Bridge, Black|Pickups
discarded pair=Dunlop 535Q Cry Baby Multi-Wah Pedal|Guitar Pedals | Effects Pedals
     */
    private int createRecord(String prodTripleIndexID,
                             SolrDocument d, SolrClient prodcatIndex, long curr) throws IOException, SolrServerException {
        String id = d.getFieldValue("id").toString();
        String h = d.getFieldValue("source_host").toString();

        //must be an English speaking country
        if (!checkHost(h))
            return 0;

        String url = d.getFieldValue("source_page").toString();
        SolrInputDocument doc = new SolrInputDocument();
        int added = 0;

        String name = null;
        if (d.getFieldValue("sg-product_name") != null)
            name = d.getFieldValue("sg-product_name").toString().replaceAll("\\s+", " ").trim();
        else if (d.getFieldValue("sg-offer_name") != null)
            name = d.getFieldValue("sg-offer_name").toString().replaceAll("\\s+", " ").trim();

        if (name != null) {
            String cat = null;
            if (d.getFieldValue("sg-product_category") != null)
                cat = d.getFieldValue("sg-product_category").toString().replaceAll("\\s+", " ").trim();
            else if (d.getFieldValue("sg-offer_category") != null)
                cat = d.getFieldValue("sg-offer_category").toString().replaceAll("\\s+", " ").trim();

            String desc = null;
            if (d.getFieldValue("sg-product_description") != null)
                desc = d.getFieldValue("sg-product_description").toString();
            else if (d.getFieldValue("sg-offer_description") != null)
                desc = d.getFieldValue("sg-offer_description").toString();

            name = cleanName(name);
            if (name==null)
                return added;

            if (cat != null)
                cat = cleanName(cat);
            if (desc != null) {
                desc = desc.replaceAll("\\s+", " ").trim();
                desc = cleanDesc(desc);
                if (desc!=null && desc.equalsIgnoreCase("NE"))
                    return added;
            }

            if (cat != null && cat.startsWith(name)) {
                cat = cat.substring(name.length()).trim();
                if (cat.length() < 3)
                    cat = null;
            }

            cat = cat == null ? "" : cat;
            desc = desc == null ? "" : desc;

            doc.setField("id", id);
            doc.setField("name", name);
            doc.setField("source_entity_index", prodTripleIndexID);
            doc.setField("category", cat);
            doc.setField("category_str", cat);
            doc.setField("desc", desc);
            doc.setField("text", desc);
            doc.setField("host", h);
            doc.setField("url", url);
            prodcatIndex.add(doc);
            added++;
        }

        return added;
    }

    public static boolean checkHost(String host) {
        for (String d : validDomains) {
            if (host.endsWith(d))
                return true;
        }
        return false;
    }

    private static String toASCII(String in) {
        String fold = in.replaceAll("[^\\p{ASCII}]", "").replaceAll("\\s+", " ").trim();
        return fold;
    }

    /**
     * implements rules to clean values in the product name and category fields
     *
     * @param value
     * @return
     */
    public static String cleanName(String value) {
        value = value.trim();
        if (value.startsWith("http"))
            return null;
        value = StringEscapeUtils.unescapeJava(value).replaceAll("\\s+", " ");
        value = StringUtils.stripAccents(value);

        String asciiValue = toASCII(value);
        String alphanumeric = asciiValue.replaceAll("[^\\p{IsAlphabetic}\\p{IsDigit}]", " ").
                replaceAll("\\s+", " ").trim().toLowerCase();
        List<String> normTokens = Arrays.asList(alphanumeric.split("\\s+"));

        if (stopwords.contains(asciiValue.toLowerCase()))
            return null;

        if (asciiValue.length() < 5)
            return null;

        if (asciiValue.startsWith(".") || asciiValue.startsWith("\\u"))
            return null;

        return asciiValue;
    }

    /**
     * implements rules to clean values in the product name and category fields
     *
     * @param value
     * @return
     */
    private String cleanDesc(String value) {
        value = StringEscapeUtils.unescapeJava(value).trim();
        if (value==null){
            return value;
        }
        value = value.replaceAll("\\s+", " ");
        value = StringUtils.stripAccents(value);
        String asciiValue = toASCII(value);

        if (stopwords.contains(asciiValue.toLowerCase()))
            return null;

        if (asciiValue.length() < 20)
            return null;
        if (asciiValue.split("\\s+").length < 10)
            return null;

        if (asciiValue.startsWith(".") || asciiValue.startsWith("\\u"))
            return null;

        TextObject textObject = textObjectFactory.forText(value);
        Optional<LdLocale> lang = languageDetector.detect(textObject);
        if (!lang.isPresent())
            return "NE";
        if (!lang.get().getLanguage().equalsIgnoreCase("en"))
            return "NE";
        return asciiValue;

    }

    public static void main(String[] args) throws IOException {
        File[] solrIndeces = new File(args[0]).listFiles();
        CoreContainer prodNCContainer = new CoreContainer(args[1]);
        prodNCContainer.load();
        SolrClient prodNameCatDescIndex = new EmbeddedSolrServer(prodNCContainer.getCore("prodcatdesc"));
        for (File f : solrIndeces) {
            String path = f.toString();

            String[] parts = path.split("/");
            String indexID = parts[parts.length - 1];
            if (path.contains(args[3]) && f.isDirectory()) {
                LOG.info("Started: " + path);
                try {
                    CoreContainer prodIndexContainer = new CoreContainer(path);
                    prodIndexContainer.load();
                    SolrClient prodTripleIndex = new EmbeddedSolrServer(prodIndexContainer.getCore("entities"));


                    ProdCatDescIndexCreator exporter = new ProdCatDescIndexCreator();
                    exporter.export(prodTripleIndex, indexID, Integer.valueOf(args[2]), prodNameCatDescIndex);
                    prodTripleIndex.close();
                } catch (Exception e) {
                    e.printStackTrace();
                    LOG.info("Failed to work in index:" + path);
                }
                LOG.info("COMPLETED for " + path);
                LOG.info("Starting next ... \n\n");
            }
        }
        //String test="Silverline Air Impact Wrench 1/2|\\n\\n  \\n    Tools\\n  \\n>\\n  \\n    Silverline Tools\\n  \\n>\\n  \\n    Air Tools\\n  \\n>\\n  \\n    Air Tools\\n  \\n\\n\\n\\n";
       /* String test = "Pasant\\u00EF\\u00BF\\u00BD King Size Condoms (singles)|Home > Condoms > Large Condoms";

        String parsed=StringEscapeUtils.unescapeHtml(test).replaceAll("\\s+"," ");
        parsed=StringUtils.stripAccents(parsed);
        System.out.println(parsed);
*/

        prodNameCatDescIndex.close();
        System.exit(0);

    }
}
