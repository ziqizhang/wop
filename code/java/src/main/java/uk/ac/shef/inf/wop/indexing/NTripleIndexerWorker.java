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
import java.util.*;
import java.util.concurrent.RecursiveTask;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

/**
 * this file reads lines of n-quads and index them accordingly to 'entities' and 'predicates' indexes
 *
 * WARNING: this assumes that same entities do not appear twice in the source data! If that's not the case, data indexed
 * may not be complete
 *
 * WARNING: you need to ensure your data are thread-safe, that is, when different parts of data are processed concurrently
 * by different threads, there will not be identical data instances written by different threads
 */

public class NTripleIndexerWorker extends RecursiveTask<Integer>{
    private SolrClient entitiesCoreClient;
    //private SolrClient predicatesCoreClient;
    private int commitBatch=5000;
    private int id;

    private long startLine;
    private long endLine;
    private boolean countLines;
    private Pattern NON_ASCII_PATTERN = Pattern.compile("[^\\\\x00-\\\\x7F]+");
    private String UNICODE_PATTERN="\\\\u[0-9A-Fa-f]{4}";

    private static final Logger LOG = Logger.getLogger(NTripleIndexerWorker.class.getName());

    private int maxTasksPerThread=1;
    private List<String> gzFiles;


    public NTripleIndexerWorker(int id,
                                SolrClient entitiesCoreClient, SolrClient predicatesCoreClient, List<String> inputGZFiles,
                                long startLine, long endLine, boolean countLines) {
        this.id=id;
        this.entitiesCoreClient = entitiesCoreClient;
        //this.predicatesCoreClient = predicatesCoreClient;
        this.startLine = startLine;
        this.endLine = endLine;
        this.gzFiles=inputGZFiles;
        this.countLines=countLines;
    }

    private Scanner setScanner(String file) throws IOException {
        InputStream fileStream = new FileInputStream(file);
        InputStream gzipStream = new GZIPInputStream(fileStream);
        Reader decoder = new InputStreamReader(gzipStream, Charset.forName("utf8"));
        Scanner inputScanner = new Scanner(decoder);
        inputScanner.useDelimiter(" .");
        LOG.info("Thread "+id+" Obtained scanner object in put file");
        return inputScanner;
    }

    protected int computeSingleWorker(List<String> gzFiles) throws IOException {
        int entityDocCount = 0;

        for(String inputGZFile: gzFiles) {
            if (this.countLines) {
                LOG.info("Thread "+id+" Counting lines begins...");
                countTotalLines(inputGZFile);
                //System.exit(0);
            }

            long lines = 0;
            String content;
            String entityID = null;
            boolean isEnglish = true;
            SolrInputDocument entityDoc = new SolrInputDocument();
            //SolrInputDocument predicateDoc = new SolrInputDocument();


            Scanner inputScanner = setScanner(inputGZFile);
            while (inputScanner.hasNextLine() && (content = inputScanner.nextLine()) != null) {
                lines++;
                //System.out.println(lines);
                if (lines < startLine)
                    continue;
                if (lines>=endLine)
                    break;
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
                    isEnglish = isEnglish(object, content.substring(lastQuote + 1));

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

                subject = subject + "|" + source;

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
                        LOG.warn(String.format("\t\tThread "+id+" failed to add doc to index at quad: %d",
                                lines, ExceptionUtils.getFullStackTrace(e)));
                    }

                    if (res) {
                        entityDocCount++;
                        if (entityDocCount % commitBatch == 0) {
                            try {
                                entitiesCoreClient.commit();
                                LOG.info(String.format("\t\tThread "+id+" completed indexing up to quad: %d and entity: %d",
                                        lines, entityDocCount));
                            } catch (SolrServerException e) {
                                LOG.warn(String.format("\t\tThread "+id+" failed to commit to server at quad: %d",
                                        lines, ExceptionUtils.getFullStackTrace(e)));
                            }
                        }
                    }

                    entityDoc = new SolrInputDocument();
                    entityID = subject;
                    entityDoc.addField("id", entityID);

                    addPredicateObject(predicate, object, entityDoc);
                    addSource(source, entityDoc);
                }

            /*
            prepare data to be written to the predicate index
             */
                //indexPredicate(predicateDoc, predicate, isEnglish);
                //predicateDoc = new SolrInputDocument();

            }


            //finally, commit the solr doc
            try {
                entitiesCoreClient.add(entityDoc);
                entitiesCoreClient.commit();
                //predicatesCoreClient.commit();
            } catch (Exception e) {
                LOG.warn(String.format("\t\tThread "+id+" failed to make the final commit at completion",
                        lines, ExceptionUtils.getFullStackTrace(e)));
            }
        }
        LOG.info("Thread "+id+" indexing completed with filtered entities="+entityDocCount);
        return entityDocCount;
    }

    /*private void indexPredicate(SolrInputDocument doc, String predicate, boolean isEnglish) {
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
    }*/

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

    private void countTotalLines(String inputGZFile) throws IOException {
        long lines = 0;
        String content;
        Scanner inputScanner = setScanner(inputGZFile);

        while (inputScanner.hasNextLine()&&(content = inputScanner.nextLine()) != null) {
            lines++;
            if (lines % 100000 == 0)
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

    @Override
    protected Integer compute() {
        if (this.gzFiles.size() > maxTasksPerThread) {
            List<NTripleIndexerWorker> subWorkers =
                    new ArrayList<>(createSubWorkers());
            for (NTripleIndexerWorker subWorker : subWorkers)
                subWorker.fork();
            return mergeResult(subWorkers);
        } else {
            try {
                return computeSingleWorker(this.gzFiles);
            } catch (IOException e) {
                LOG.warn(String.format("\t\tunable to read input gz file: %s, \n %s",
                        this.gzFiles.toString(), ExceptionUtils.getFullStackTrace(e)));
                return 0;
            }
        }
    }


    protected List<NTripleIndexerWorker> createSubWorkers() {
        List<NTripleIndexerWorker> subWorkers =
                new ArrayList<>();

        boolean b = false;
        List<String> splitTask1 = new ArrayList<>();
        List<String> splitTask2 = new ArrayList<>();
        for (String s: gzFiles) {
            if (b)
                splitTask1.add(s);
            else
                splitTask2.add(s);
            b = !b;
        }

        NTripleIndexerWorker subWorker1 = createInstance(splitTask1, this.id+1);
        NTripleIndexerWorker subWorker2 = createInstance(splitTask2, this.id+2);

        subWorkers.add(subWorker1);
        subWorkers.add(subWorker2);

        return subWorkers;
    }

    /**
     * NOTE: classes implementing this method must call setHashtagMap and setMaxPerThread after creating your object!!
     * @param splitTasks
     * @param id
     * @return
     */
    protected NTripleIndexerWorker createInstance(List<String> splitTasks, int id){
        NTripleIndexerWorker indexer = new NTripleIndexerWorker(id,
                entitiesCoreClient, null,splitTasks,
                startLine, endLine, countLines);
        return indexer;
    }
    /*{
        return new NTripleIndexerApp(id, this.solrClient, splitTasks, maxTasksPerThread, outFolder);
    }*/

    protected int mergeResult(List<NTripleIndexerWorker> workers) {
        Integer total = 0;
        for (NTripleIndexerWorker worker : workers) {
            total += worker.join();
        }
        return total;
    }
}
