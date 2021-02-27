package tools
// I don't really know Scala. I am a Haskeller

import com.babelscape.util.{POS, UniversalPOS}
import scala.collection.JavaConverters._
import it.uniroma1.lcl.babelnet._
import it.uniroma1.lcl.babelnet.data.{BabelGloss, BabelSenseSource}
import it.uniroma1.lcl.jlt.util.Language
import java.{util => ju}
import it.uniroma1.lcl.kb.ResourceID

class BabelNetLexeme(val id: ResourceID,
                     val lexeme: String,
                     val pos: POS,
                     val synsets: ju.List[BabelGloss],
                     val relatedWords: ju.List[BabelSynsetRelation])

object BabelNetBridge {
  private def linguistic(source: BabelSenseSource): Boolean = {
    source.isFromBabelNet || source.isFromWiktionary ||
      source.isFromWordAtlas || source.isFromWordNet || source.isFromOmegaWiki ||
      source.isFromAnyWordnet
  }

  def getSynsetsForLexeme(lexeme: String, pos: String): ju.List[BabelNetLexeme] = {
    val bn = BabelNet.getInstance
    val upos = UniversalPOS.valueOf(pos.toUpperCase)

    val query = new BabelNetQuery.Builder(lexeme)
      .from(Language.EN)
      .POS(upos)
      .build

    val byl = bn.getSynsets(query)

    // Get synsets that have at least one linguistic source
    val senses = byl.asScala.filter(_.getSenseSources.asScala.exists(linguistic))

    (
      for {
        sense <- senses
        glosses = sense.getGlosses().asScala.filter(_.getLanguage == Language.EN)
        edges = sense.getOutgoingEdges
        babelnetLexeme = new BabelNetLexeme(sense.getID, lexeme, upos, glosses.asJava, edges)
      } yield babelnetLexeme
    ).asJava
  }
  def main(args: Array[String]): Unit = {
    for (home <- getSynsetsForLexeme("home", "verb").asScala)
      {
        println(home.id, home.lexeme, home.synsets)
      }
  }
}
