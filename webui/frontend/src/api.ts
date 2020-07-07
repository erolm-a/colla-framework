const API_URL = "http://knowledge-glue-webui-jeffstudentsproject.ida.dcs.gla.ac.uk/api";

type POS = "noun" | "verb" | "adjective" | "pronoun"
interface ExpressionDefinition {
    pos: POS
}

async function define_expression(): Promise<ExpressionDefinition[]>
{
    const definition_api = API_URL + "/word/";
    const response = await fetch(definition_api);

    if (response.ok)
        return response.json();
    else {
        console.warn(response.statusText);
        return [];
    }
}