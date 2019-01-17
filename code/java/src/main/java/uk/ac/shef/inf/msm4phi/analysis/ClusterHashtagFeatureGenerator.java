package uk.ac.shef.inf.msm4phi.analysis;

import com.opencsv.CSVWriter;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.embedded.EmbeddedSolrServer;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import org.apache.solr.core.CoreContainer;
import uk.ac.shef.inf.msm4phi.Util;


import java.io.IOException;
import java.util.*;

public class ClusterHashtagFeatureGenerator {
    private static final int resultBatchSize = 10000;
    private static final Logger LOG = Logger.getLogger(ClusterHashtagFeatureGenerator.class.getName());

    public static void main(String[] args) throws IOException {
        CoreContainer solrContainer = new CoreContainer(args[0]);
        solrContainer.load();

        SolrClient tweetSolrClient = new EmbeddedSolrServer(solrContainer.getCore("tweets"));

        ClusterHashtagFeatureGenerator ufg = new ClusterHashtagFeatureGenerator();
        ufg.process(args[1],args[2],tweetSolrClient);

        tweetSolrClient.close();

        System.exit(0);
    }

    private void process(String hashtagFile, String outFile,
                         SolrClient tweetCore) throws IOException {
        Map<String, String> tag2diseaseInput = Common.createInverseHashtagMap(hashtagFile);
        LOG.info("Calculating hashtags in multiple disease communities...");
        Map<String, Set<String>> tagsAndDiseases = Common.findTagsAndDiseases(tweetCore, tag2diseaseInput, LOG, resultBatchSize);
        Map<String, Set<String>> selected=selectAndPrintStats(tagsAndDiseases);

        Set<String> validDiseases = new HashSet<>();
        for(Set<String> values : selected.values())
            validDiseases.addAll(values);
        List<String> orderedDiseases = new ArrayList<>(validDiseases);
        List<String> orderedTags = new ArrayList<>(selected.keySet());
        Collections.sort(orderedTags);

        LOG.info("\nCalculating feature matrix...");
        Matrix m = generateFeatures(tweetCore, selected, orderedDiseases, orderedTags);

        LOG.info("\nOutputing data...");
        CSVWriter csvWriter = null;

        csvWriter = Util.createCSVWriter(outFile);
        String[] header = new String[orderedTags.size()+1];
        header[0]="";
        for(int i=0; i<orderedTags.size(); i++)
            header[i+1]=orderedTags.get(i);
        csvWriter.writeNext(header);
        for (int r=0; r<m.numRows();r++){
            String[] row = new String[orderedTags.size()+1];
            row[0]=orderedDiseases.get(r);
            for(int c=0;c<m.numColumns(); c++)
                row[c]=String.valueOf(m.get(r,c));
            csvWriter.writeNext(row);
        }
        csvWriter.close();
    }

    private Matrix generateFeatures(SolrClient tweetCore,
                                    Map<String, Set<String>> tag2diseases,
                                    List<String> orderedDiseases, List<String> orderedTags) {

        Matrix m = new LinkedSparseMatrix(orderedDiseases.size(), tag2diseases.size());

        SolrQuery query = new SolrQuery();
        query.setQuery("*:*");
        query.setStart(0);
        query.setRows(resultBatchSize);
        long tweets = 0;
        boolean stop = false;
        while (!stop) {
            QueryResponse qr = null;
            try {
                qr = Util.performQuery(query, tweetCore);
                if (qr != null)
                    tweets = qr.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        tweets, query.getStart(), query.getStart() + query.getRows()));
                for (SolrDocument d : qr.getResults()) {
                    Collection<Object> hashtags = d.getFieldValues("entities_hashtag");
                    if(hashtags==null)
                        continue;
                    for (Object h : hashtags) {
                        String tag=h.toString();
                        Set<String> dis = tag2diseases.get(tag);
                        if (dis == null)
                            continue;

                        for(String d_: dis) {
                            int diseaseIndex = orderedDiseases.indexOf(d_);
                            if (diseaseIndex < 0)
                                continue;

                            int tagIndex = orderedTags.indexOf(tag);
                            m.set(diseaseIndex, tagIndex, m.get(diseaseIndex, tagIndex) + 1);
                        }
                    }

                }
            } catch (Exception e) {
                LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        query.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = query.getStart() + query.getRows();
            if (curr < tweets)
                query.setStart(curr);
            else
                stop = true;
        }
        return m;
    }

    private Map<String, Set<String>> selectAndPrintStats(Map<String, Set<String>> tag2diseases){
        Map<String, Set<String>> selected = new HashMap<>();
        List<Double> diseaseNum = new ArrayList<>();
        for(Map.Entry<String, Set<String>> en:tag2diseases.entrySet()){
            if(en.getValue().size()>1){
                selected.put(en.getKey(), en.getValue());
                diseaseNum.add((double)en.getValue().size());
            }
        }

        Collections.sort(diseaseNum);
        Collections.reverse(diseaseNum);
        List<String> tags = new ArrayList<>(selected.keySet());
        tags.sort((s, t1) ->
                Integer.compare(selected.get(t1).size(), selected.get(s).size()));
        double[] values = ArrayUtils.toPrimitive(diseaseNum.toArray(new Double[0]));

        System.out.println(String.format("%d out of %d has >1 disease communities. For these, " +
                        "mean=%.2f, max=%.2f, median=%.2f",tag2diseases.size(), selected.size(),
                StatUtils.mean(values), StatUtils.max(values), StatUtils.percentile(values, 0.5)));
        return selected;
    }


}
