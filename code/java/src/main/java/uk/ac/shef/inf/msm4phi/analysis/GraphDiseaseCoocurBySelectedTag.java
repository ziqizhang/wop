package uk.ac.shef.inf.msm4phi.analysis;

import com.opencsv.CSVWriter;
import no.uib.cipr.matrix.Matrix;
import no.uib.cipr.matrix.sparse.CompRowMatrix;
import no.uib.cipr.matrix.sparse.LinkedSparseMatrix;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang.exception.ExceptionUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.log4j.Logger;
import org.apache.solr.client.solrj.SolrClient;
import org.apache.solr.client.solrj.SolrQuery;
import org.apache.solr.client.solrj.response.QueryResponse;
import org.apache.solr.common.SolrDocument;
import uk.ac.shef.inf.msm4phi.Util;

import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * co-occureence of two diseases = # unique hashtags in both D1 and D2 / #unique hastags in either D1 and D2.
 *
 * only legit disease tags selected from symplur are considered
 *
 */
public class GraphDiseaseCoocurBySelectedTag {

    private static final int resultBatchSize=10000;
    private static final Logger LOG = Logger.getLogger(GraphDiseaseCoocurByTag.class.getName());

    private Set<String> exclusion = new HashSet<>();

    public GraphDiseaseCoocurBySelectedTag(String exclusionFile) throws IOException {
        if (exclusionFile!=null)
            exclusion =Common.readExcludeList(exclusionFile,Integer.MAX_VALUE);
    }

    void process(SolrClient tweetCore, String hashtagFile, String outFile) throws IOException {
        Map<String, List<String>> disease2tagsInput=Util.readHashtags(new File(hashtagFile));
        Pair<Matrix, List<String>> output=calculateSimilarityMatrix(disease2tagsInput,tweetCore);
        Common.saveMatrixData(output.getLeft(), output.getRight(), outFile);
    }

    private Pair<Matrix, List<String>> calculateSimilarityMatrix(Map<String, List<String>> disease2tagsInput,
                                                                 SolrClient tweetCore){
        Set<String> legitDiseaseTags = new HashSet<>();
        for (List<String> sets: disease2tagsInput.values()){
            legitDiseaseTags.addAll(sets);
        }
        Map<String, Set<String>> disease2AllTags=new HashMap<>();
        for(Map.Entry<String, List<String>> entry : disease2tagsInput.entrySet()){
            LOG.info("Processing disease="+entry.getKey());
            Set<String> allTags = collectHashtagsOfDisease(tweetCore, entry.getValue().toArray(new String[0]));
            allTags.retainAll(legitDiseaseTags);

            if (allTags!=null &&allTags.size()>0)
                disease2AllTags.put(entry.getKey(), allTags);
        }

        List<String> filteredDiseases = new ArrayList<>(disease2AllTags.keySet());
        Collections.sort(filteredDiseases);
        Matrix m = new LinkedSparseMatrix(filteredDiseases.size(), filteredDiseases.size());

        double max=0.0;
        for(int i=0; i<filteredDiseases.size();i++){
            String diseaseA=filteredDiseases.get(i);
            Set<String> tagsA =disease2AllTags.get(diseaseA);
            System.out.println(i);
            for(int j=i+1; j<filteredDiseases.size();j++){
                String diseaseB=filteredDiseases.get(j);
                Set<String> tagsB =disease2AllTags.get(diseaseB);

                Collection<String> inter=CollectionUtils.intersection(tagsA, tagsB);
                double sim = (double)inter.size()/CollectionUtils.union(tagsA,tagsB).size();
                if (sim>max)
                    max=sim;
                m.set(i,j,sim);
                m.set(j,i, sim);
            }
        }

        normalize(m, max);

        return new ImmutablePair<>(m, filteredDiseases);
    }

    private void normalize(Matrix m, double max) {
        for(int r=0;r<m.numRows();r++){
            for(int c=0;c<m.numColumns();c++)
                m.set(r,c, m.get(r,c)/max);
        }
    }

    private Set<String> collectHashtagsOfDisease(SolrClient tweetCore,
                                                 String... diseaseTags){
        Set<String> tags =new HashSet<>();
        SolrQuery q=Util.createQueryTweetsOfHashtags(resultBatchSize, diseaseTags);
        boolean stop = false;
        while (!stop) {
            QueryResponse res = null;
            long total = 0;
            try {
                res = Util.performQuery(q, tweetCore);
                if (res != null)
                    total = res.getResults().getNumFound();
                //update results
                LOG.info(String.format("\t\ttotal results of %d, currently processing from %d to %d...",
                        total, q.getStart(), q.getStart() + q.getRows()));
                for (SolrDocument d : res.getResults()) {
                    Collection<Object> hashtags =d.getFieldValues("entities_hashtag");
                    if (hashtags!=null){
                        for (Object o : hashtags){
                            String ht = o.toString();
                            if (ht.length()>1)
                                tags.add(ht);
                        }
                    }
                }

            } catch (Exception e) {
                LOG.warn(String.format("\t\tquery %s caused an exception: \n\t %s \n\t trying for the next query...",
                        q.toQueryString(), ExceptionUtils.getFullStackTrace(e)));
            }

            int curr = q.getStart() + q.getRows();
            if (curr < total)
                q.setStart(curr);
            else
                stop = true;

        }
        tags.removeAll(exclusion);
        return tags;
    }
}
