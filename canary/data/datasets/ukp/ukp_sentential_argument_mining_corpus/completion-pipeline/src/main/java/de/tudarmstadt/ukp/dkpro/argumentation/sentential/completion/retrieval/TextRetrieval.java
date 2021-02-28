package de.tudarmstadt.ukp.dkpro.argumentation.sentential.completion.retrieval;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.SocketTimeoutException;
import java.net.URL;
import java.nio.charset.StandardCharsets;

import org.apache.commons.io.IOUtils;
import org.apache.http.HttpStatus;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.html.BoilerpipeContentHandler;
import org.apache.tika.sax.BodyContentHandler;

/**
 * Class to crawl a web page, and return the contents in text format, after
 * boiler plater removal.
 * 
 * @author Christian Stab and Debanjan Chaudhuri
 *
 */
public class TextRetrieval {

	private static String userAgent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36";

	/**
	 * 
	 * @param webArchvURL,
	 *            not null
	 * @param origURL,
	 *            not null
	 * @param retries,
	 *            not null
	 * @return the text from the web archive url
	 * @throws Exception
	 */
	public static String getContentFromURL(String webArchvURL, String origURL, int retries) throws Exception {

		String result = "";

		HttpResponse responseFromUrl = contentRetrieval(webArchvURL);
		if (responseFromUrl.statusCode == HttpStatus.SC_OK) {
			return responseFromUrl.htmlContent;
		} else {
			int retryCount = retries;
			Boolean success = false;

			// retry if first attempt didn't work
			while (retryCount > 0) {
				HttpResponse responseRetry = contentRetrieval(webArchvURL);
				// System.out.println("Retrying .... for status code:" +
				// responseRetry.statusCode + " for url:" + webArchvURL);
				if (responseRetry.statusCode == HttpStatus.SC_OK) {
					result = responseRetry.htmlContent;
					success = true;
					// System.out.println("Success from retrying, going to next fetch!");
					break;
				} else {
					retryCount = retryCount - 1;
				}
			}

			if (!success) {
				// fetch HTML from original URL
				System.out.println("  Warning: Not able to retrieve text from '" + webArchvURL
						+ "' using original url instead ('" + origURL + "')");
				result = contentRetrieval(origURL).htmlContent;
			}
		}

		return (result);
	}

	/**
	 * Remove tool bar code from web archive html Page
	 * 
	 * @param html
	 * @return
	 */
	private static String removeArchiveToolBar(String html) {
		// if html page is from web archive remove tool bar code
		if (html.indexOf("<!-- BEGIN WAYBACK TOOLBAR INSERT -->") > -1) {
			// System.out.println("##### Remove Waybackmachine studff");
			String tmp = html.substring(0, html.indexOf("<!-- BEGIN WAYBACK TOOLBAR INSERT -->"));
			tmp = tmp + html.substring(html.indexOf("<!-- END WAYBACK TOOLBAR INSERT -->") + 35, html.length());
			html = tmp;
		}
		return html;
	}

	/**
	 * Method for getting only the content of the URL, will return no content if
	 * response is not 200
	 * 
	 * @param url,
	 *            not null
	 * @return the text from the url, after boilerplate removal.
	 * @throws Exception
	 */
	public static HttpResponse contentRetrieval(String url) throws Exception {
		// System.out.println("Retrieve text from '" + url + "'");
		StringBuilder htmlcontent = new StringBuilder();
		HttpResponse response = new HttpResponse();
		try {
			URL u = new URL(url);
			HttpURLConnection con = (HttpURLConnection) u.openConnection();
			// con.setRequestProperty("User-Agent", "Mozilla/5.0 (Windows NT 10.0; WOW64)
			// AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.110 Safari/537.36");
			con.setRequestProperty("User-Agent", userAgent);

			int statusCode = con.getResponseCode();

			if (statusCode == HttpStatus.SC_OK) {

				InputStream stream = con.getInputStream();

				// Boiler plate removal using Apache Tika
				AutoDetectParser parser = new AutoDetectParser();
				BodyContentHandler handler = new BodyContentHandler(-1);
				ParseContext context = new ParseContext();
				Metadata metadata = new Metadata();
				// parser.parse(stream, new BoilerpipeContentHandler(handler), metadata,
				// context);
				if (url.toLowerCase().endsWith(".pdf")) {
					parser.parse(stream, new BoilerpipeContentHandler(handler), metadata, context);
				} else {
					String htmlString = IOUtils.toString(stream, StandardCharsets.UTF_8);
					htmlString = removeArchiveToolBar(htmlString);
					parser.parse(new ByteArrayInputStream(htmlString.getBytes()), new BoilerpipeContentHandler(handler),
							metadata, context);
				}

				for (String substring : handler.toString().split("\n")) {
					htmlcontent.append(substring + "\n");
				}
			} else {
				// handle redirect
				if (statusCode == HttpURLConnection.HTTP_MOVED_TEMP || statusCode == HttpURLConnection.HTTP_MOVED_PERM
						|| statusCode == HttpURLConnection.HTTP_SEE_OTHER) {
					// get redirect url from "location" header field
					String newUrl = con.getHeaderField("Location");
					return contentRetrieval(newUrl);
				} else {
					htmlcontent.append("");
				}
			}

			response.statusCode = statusCode;

		} catch (SocketTimeoutException e) {
			System.out.println("  Warning: Unable to load text from '" + url + "'");
		}

		response.htmlContent = htmlcontent.toString();

		return response;
	}

}
