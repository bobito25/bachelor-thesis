

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b d)
(on c f)
(on d e)
(on e a)
(on f b)
(clear c)
)
(:goal
(and
(on e c))
)
)


