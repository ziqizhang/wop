package uk.ac.shef.inf.wop.app;

import java.util.HashSet;
import java.util.Set;

class ProdDescCorpusForBert_ConcurrentSet {

    private Set<String> results=new HashSet<>();

    public synchronized void add(String v){
        results.add(v);
    }

    public synchronized boolean contains(String v){
        return results.contains(v);
    }
}
