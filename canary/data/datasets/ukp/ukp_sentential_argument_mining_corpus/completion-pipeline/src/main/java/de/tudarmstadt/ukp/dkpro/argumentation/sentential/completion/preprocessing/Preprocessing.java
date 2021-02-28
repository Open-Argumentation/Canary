package de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.preprocessing;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.pipeline.SimplePipeline.runPipeline;

import java.util.Collection;
import java.util.HashMap;

import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;

import de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.tools.Hash;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.stanfordnlp.StanfordSegmenter;

public class Preprocessing {

	/**
	 * This method takes texts, a string and the language, and returns a hashmap of
	 * sentence hash as keys and tokenized sentences as values
	 * 
	 * @param str
	 * @param lang
	 * @return
	 */
	public static HashMap<String, String> extractSentences(String str, String lang) {
		HashMap<String, String> retrievedSentences = new HashMap<String, String>();

		try {
			JCas jcas = JCasFactory.createJCas();
			jcas.setDocumentText(str);
			DocumentMetaData meta = DocumentMetaData.create(jcas);
			meta.setLanguage(lang);

			// preprocessing
			AnalysisEngineDescription preprocessing = createEngineDescription(
					createEngineDescription(StanfordSegmenter.class));

			// run Pipeline
			runPipeline(jcas, preprocessing);

			// collect sentences
			Collection<Sentence> sentences = JCasUtil.select(jcas, Sentence.class);
			for (Sentence s : sentences) {
				String sentence = getTokenizedSentence(s);
				retrievedSentences.put(Hash.get(sentence), sentence);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		return retrievedSentences;
	}

	private static String getTokenizedSentence(Sentence s) {
		String result = "";
		Collection<Token> tokens = JCasUtil.selectCovered(Token.class, s);
		for (Token t : tokens) {
			result = result + t.getCoveredText() + " ";
		}
		return trimSentence(result);
	}

	private static String trimSentence(String s) {
		return s.replaceAll("\\[\\s[0-9]*\\s\\]", "").trim().replaceAll("\t", "").replaceAll(" +", " ");
	}

}
