<template>
    <div class="container" style="text-align: left">
        <b-breadcrumb :items="path">
        </b-breadcrumb>
        <b-list-group v-for="predicate in predicates" :key="predicate.name">
            <k-g-collapse :property="predicate" />
        </b-list-group> 
    </div>
</template>

<script lang="ts">
import { Component, Watch, Vue } from 'vue-property-decorator'
import {searchKGItem} from '../api';
import KGCollapse from '@/components/KGCollapse.vue';
import { Route } from 'vue-router';

@Component({
    components: {
        KGCollapse
    }
})
export default class KGVisualiser extends Vue
{
    private path: Array<{text: string; to: string}> = []
    private predicates: Array<Record<string, any>> = []

    async beforeRouteUpdate(to: Route, from: Route, next: (() => void)) {
        const path = this.path;
        console.log(this.path);
        if(path.length >= 2 && to.params.entity == path[path.length-2].text)
        {
            console.log("Going back")
            path.pop();
        }
        else
        {
            console.log("Going forward")
            path.push({text: to.params.entity, to: to.path})
        }
        
        await this.updatePredicates(to);
        next();
    }

    private async updatePredicates (route: Route) {
        const entityId = route.params.entity;
        console.log("predicates: " + this.predicates);
        try {
            console.log(this.$route.params.entity);
            const entity = (await searchKGItem(entityId))[0];
            const predicateObjects = Object.entries(entity);

            while(this.predicates.length)
                this.predicates.pop();
            
            for(const [predicate, nested] of predicateObjects)
            {
                const newLeaf = {name: predicate, values: nested};
                this.predicates.push(newLeaf);
            }
        }
        catch(error)
        {
            console.error(error);
        }             
    }

    @Watch('this.$route')
    onRouteChange(value: Route) {
        this.updatePredicates(value);
    }

    mounted() {
        this.path.push({text: this.$route.params.entity, to: this.$route.path});
        this.updatePredicates(this.$route);
    }

}
</script>