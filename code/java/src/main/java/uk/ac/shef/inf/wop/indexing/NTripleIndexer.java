package uk.ac.shef.inf.wop.indexing;

import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrServerException;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.common.SolrInputDocument;
import org.apache.solr.core.CoreContainer;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

/**
 * this file reads lines of n-quads and index them accordingly to 'entities' and 'predicates' indexes
 *
 * WARNING: this assumes that same entities do not appear twice in the source data! If that's not the case, data indexed
 * may not be complete
 */

public class NTripleIndexer {
    private SolrClient entitiesCoreClient;
    private SolrClient predicatesCoreClient;

    private long startLine;
    private long endLine = Long.MAX_VALUE;
    private Pattern NON_ASCII_PATTERN = Pattern.compile("[^\\\\x00-\\\\x7F]+");
    private String UNICODE_PATTERN="\\\\u[0-9A-Fa-f]{4}";

    private String inputGZFile;
    private static final Logger LOG = Logger.getLogger(NTripleIndexer.class.getName());


    public static void main(String[] args) throws IOException {

        CoreContainer solrContainer = new CoreContainer(args[1]);
        solrContainer.load();

        SolrClient entitiesCoreClient = new EmbeddedSolrServer(solrContainer.getCore("entities"));
        SolrClient predicatesCoreClient= new EmbeddedSolrServer(solrContainer.getCore("predicates"));

        LOG.info("Initialisation completed.");
        NTripleIndexer indexer = new NTripleIndexer(entitiesCoreClient, predicatesCoreClient,args[0],
                0, Long.MAX_VALUE);
        indexer.startIndexing(true);
        entitiesCoreClient.close();
        predicatesCoreClient.close();
        System.exit(0);
    }



    public NTripleIndexer(SolrClient entitiesCoreClient, SolrClient predicatesCoreClient, String inputGZFile,
                          long startLine, long endLine) {
        this.entitiesCoreClient = entitiesCoreClient;
        this.predicatesCoreClient = predicatesCoreClient;
        this.startLine = startLine;
        this.endLine = endLine;
        this.inputGZFile=inputGZFile;
    }

    private Scanner setScanner(String file) throws IOException {
        InputStream fileStream = new FileInputStream(file);
        InputStream gzipStream = new GZIPInputStream(fileStream);
        Reader decoder = new InputStreamReader(gzipStream, Charset.forName("utf8"));
        Scanner inputScanner = new Scanner(decoder);
        inputScanner.useDelimiter(" .");
        return inputScanner;
    }

    public void startIndexing(boolean countLines) throws IOException {
        if (countLines) {
            countTotalLines();
        }

        long lines = 0;
        String content;
        String entityID = null;
        boolean isEnglish = true;
        SolrInputDocument entityDoc = new SolrInputDocument();
        SolrInputDocument predicateDoc = new SolrInputDocument();
        int entityDocCount = 0;

        Scanner inputScanner = setScanner(inputGZFile);
        while (inputScanner.hasNextLine() && (content = inputScanner.nextLine()) != null) {
            lines++;
            //System.out.println(lines);
            if (lines < startLine)
                continue;

            /*
            Parsing the s, p, o, and source
             */
            String subject = null, predicate = null, object = null, source = null;

            //do we have data literal?
            int firstQuote = content.indexOf("\"");
            int lastQuote = content.lastIndexOf("\"");
            //if yes...
            if (firstQuote != -1 && lastQuote != -1 && lastQuote > firstQuote) {
                object = content.substring(firstQuote + 1, lastQuote).trim();
                isEnglish = isEnglish(object, content.substring(lastQuote+1));

                String[] s_and_p = content.substring(0, firstQuote).trim().split("\\s+");
                if (s_and_p.length < 2)
                    continue;
                subject = trimBrackets(s_and_p[0]);
                predicate = trimBrackets(s_and_p[1]);

                source = content.substring(lastQuote + 1);
                int trim = source.indexOf(" ");
                source = trimBrackets(source.substring(trim + 1, source.lastIndexOf(" ")));
            } else { //if no, all four parts of the quad are URIs
                String[] parts = content.split("\\s+");
                if (parts.length < 4)
                    continue;
                subject = trimBrackets(parts[0]);
                predicate = trimBrackets(parts[1]);
                object = trimBrackets(parts[2]);
                source = trimBrackets(parts[3]);
            }

            subject=subject+"|"+source;

            /*
            prepare data to be written to the entity index
             */
            if (entityID == null) { //initiate a new solr doc
                entityID = subject;
                entityDoc.addField("id", entityID);
                addPredicateObject(predicate, object, entityDoc);
                addSource(source, entityDoc);
            } else if (entityID.equalsIgnoreCase(subject)) {//continue to update the solr doc
                addPredicateObject(predicate, object, entityDoc);
            } else {//we have encountered a different entity, index the prev solr doc, and start a new solr doc
                boolean res = false;
                try {
                    res = indexEntity(entityDoc, isEnglish, source);
                } catch (SolrServerException e) {
                    LOG.warn(String.format("\t\tfailed to add doc to index at line: %d",
                            lines, ExceptionUtils.getFullStackTrace(e)));
                }

                if (res) {
                    entityDocCount++;
                    if (entityDocCount % 500 == 0) {
                        try {
                            entitiesCoreClient.commit();
                            LOG.info(String.format("\t\tcompleted indexing up to line: %d",
                                    lines));
                        } catch (SolrServerException e) {
                            LOG.warn(String.format("\t\tfailed to commit to server at line: %d",
                                    lines, ExceptionUtils.getFullStackTrace(e)));
                        }
                    }
                }

                entityDoc=new SolrInputDocument();
                entityID = subject;
                entityDoc.addField("id", entityID);

                addPredicateObject(predicate, object, entityDoc);
                addSource(source, entityDoc);
            }

            /*
            prepare data to be written to the predicate index
             */
            indexPredicate(predicateDoc,predicate, isEnglish);
            predicateDoc = new SolrInputDocument();

        }


        //finally, commit the solr doc
        try{
            entitiesCoreClient.add(entityDoc);
            entitiesCoreClient.commit();
            predicatesCoreClient.commit();
        }catch (Exception e){
            LOG.warn(String.format("\t\tfailed to make the final commit at completion",
                    lines, ExceptionUtils.getFullStackTrace(e)));
        }
        LOG.info("indexing completed with filtered entities="+entityDocCount);
    }

    private void indexPredicate(SolrInputDocument doc, String predicate, boolean isEnglish) {
        doc.addField("id", predicate);
        URI u = null;
        try {
            u = new URI(predicate);
            doc.addField("host", u.getHost());
            doc.addField("paths", u.getPath());
            doc.addField("as_text", u.toASCIIString().replaceAll("[^a-zA-Z0-9]", " ").trim());
            predicatesCoreClient.add(doc);
        } catch (Exception e) {
            LOG.warn(String.format("\t\tencountered illegal URI when trying to parse (during predicate indexing): %s",
                    predicate, ExceptionUtils.getFullStackTrace(e)));
        }
    }

    /**
     * write the prepared solr document to the index only if the language is English. if already exists
     *  and language is not english, delete it
     * @param doc
     * @param isEnglish
     * @param source
     * @return
     * @throws IOException
     * @throws SolrServerException
     */
    private boolean indexEntity(SolrInputDocument doc, boolean isEnglish, String source) throws IOException, SolrServerException {
        if (isEnglish){
            entitiesCoreClient.add(doc);
            return true;
        }
        else{ //try delete if already added to index
            entitiesCoreClient.deleteById(doc.getFieldValue("id").toString());
        }
        return false;
    }

    /**
     * process the triple's predicate and object values, and write data to corresponding fields
     * @param predicate
     * @param object
     * @param doc
     */
    private void addPredicateObject(String predicate, String object, SolrInputDocument doc) {
        Map<String,Object> fieldModifier = new HashMap<>(1);
        fieldModifier.put("set",object);

        if (predicate.equalsIgnoreCase("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
            addOrMergeField(doc,"rdfs_type", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/name"))
            addOrMergeField(doc,"sg-product_name", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/description"))
            addOrMergeField(doc,"sg-product_description", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/brand"))
            addOrMergeField(doc,"sg-product_brand", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/category"))
            addOrMergeField(doc,"sg-product_category", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/Offer/name"))
            addOrMergeField(doc,"sg-offer_name", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/Offer/description"))
            addOrMergeField(doc,"sg-offer_description", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/Offer/category"))
            addOrMergeField(doc,"sg-offer_category", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/ListItem/name"))
            addOrMergeField(doc,"sg-listitem_name", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://schema.org/ListItem/description"))
            addOrMergeField(doc,"sg-listitem_description", fieldModifier);
        else if (predicate.equalsIgnoreCase("http://data-vocabulary.org/Breadcrumb/title") ||
                predicate.equalsIgnoreCase("http://data-vocabulary.org/Breadcrumb/name"))
            addOrMergeField(doc,"sg-breadcrumb_title", fieldModifier);
        else {
            addOrMergeField(doc,predicate+"_t", fieldModifier);

        /*if (predicate.equalsIgnoreCase("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))
            doc.addField("rdfs_type", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/name"))
            doc.addField("sg-product_name", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/description"))
            doc.addField("sg-product_description", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/brand"))
            doc.addField("sg-product_brand", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/Product/category"))
            doc.addField("sg-product_category", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/Offer/name"))
            doc.addField("sg-offer_name", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/Offer/description"))
            doc.addField("sg-offer_description", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/Offer/category"))
            doc.addField("sg-offer_category", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/ListItem/name"))
            doc.addField("sg-listitem_name", object);
        else if (predicate.equalsIgnoreCase("http://schema.org/ListItem/description"))
            doc.addField("sg-listitem_description", object);
        else if (predicate.equalsIgnoreCase("http://data-vocabulary.org/Breadcrumb/title") ||
                predicate.equalsIgnoreCase("http://data-vocabulary.org/Breadcrumb/name"))
            doc.addField("sg-breadcrumb_title", object);
        else {
            doc.addField(predicate+"_t", object);*/
        }
    }

    private void addOrMergeField(SolrInputDocument doc, String field, Map<String,Object> fieldModifier){
        Object existingValue = doc.getFieldValue(field);
        if (existingValue!=null){
            fieldModifier.put("set", fieldModifier.get("set").toString()+" "+existingValue.toString());
            doc.removeField(field);
        }
        doc.addField(field, fieldModifier);
    }

    /**
     * process the triple'ssource, and write data to corresponding fields
     */
    private void addSource(String sourceURI, SolrInputDocument doc){
        Map<String,Object> fieldModifier = new HashMap<>(1);
        fieldModifier.put("set",sourceURI);
        doc.addField("source_page", fieldModifier);
        URI u = null;
        try {
            u = new URI(sourceURI);
            fieldModifier = new HashMap<>(1);
            fieldModifier.put("set",u.getHost());
            doc.addField("source_host", fieldModifier);
        } catch (URISyntaxException e) {
            LOG.warn(String.format("\t\tencountered illegal URI when trying to parse: %s",
                    sourceURI, ExceptionUtils.getFullStackTrace(e)));
        }
    }

    private void countTotalLines() throws IOException {
        long lines = 0;
        String content;
        Scanner inputScanner = setScanner(inputGZFile);
        while (inputScanner.hasNextLine()&&(content = inputScanner.nextLine()) != null) {
            lines++;
            if (lines % 1000000 == 0)
                System.out.println(new Date() + ": " + lines);
        }
        System.out.println(new Date() + " counting completed: " + lines);
    }

    private boolean isEnglish(String dataLiteral, String langSuffix) {
        langSuffix = langSuffix.toLowerCase();
        if (langSuffix.startsWith("@")) {
            if (langSuffix.startsWith("@en"))
                return true;
            else
                return false;
        } else {
            if (dataLiteral.length()<10)
                return true;
            String clean = dataLiteral.replaceAll(UNICODE_PATTERN,"").trim();
            boolean r = clean.length()>dataLiteral.length()/2;
           /* Matcher matcher = NON_ASCII_PATTERN.matcher(dataLiteral);
            int count = 0;
            while (matcher.find())
                count++;
            boolean r= count > dataLiteral.length()/2;*/
            return r;
        }
    }

    private String trimBrackets(String line) {
        if (line.startsWith("<"))
            line = line.substring(1);
        if (line.endsWith(">"))
            line = line.substring(0, line.length() - 1);
        return line;
    }
}
